// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The CPU side of the summits story: reading back the compact summit list, applying the drop
 * threshold to it, classing survivors by drop, matching survivors to the OpenStreetMap catalogue,
 * counting what the window edge removes and summarising peak-snap outcomes. Pure functions over
 * typed arrays; no GPU and no DOM.
 */

import type {AlpsTerrain} from './b14b-terrain';
import {toFloat32, toUint32} from './b14b-terrain';
import type {AlpsContext} from './terrain-places';
import {NAME_MATCH_METERS, SNAP_STATUS_IDS, getSnapStatusId} from './summits-style';
import {SUMMIT_DROP_CLASSES, type SnapStatusId} from './terrain-palettes';

/** One disc maximum found by the GPU: a cell that is the highest of its disc. */
export type SummitCandidate = {
  column: number;
  row: number;
  /** DEM elevation of the cell on the analysis grid, metres. */
  elevation: number;
  /** Drop to the highest ring cell, metres (a lower bound of prominence). */
  drop: number;
};

/** The compact list of one summit run, as read back. */
export type SummitList = {
  /** Candidates in the list (at most the capacity). */
  candidates: SummitCandidate[];
  /** Candidates the GPU found, listed or not. */
  total: number;
  /** True when the list was full and more candidates exist than it holds. */
  listOverflow: boolean;
  /** True when the requested radius exceeded the compile-time search bound. */
  radiusClamped: boolean;
};

/**
 * Reads the bytes of a summit readback: `count, total, overflow, clamped` (four uint32), then the
 * ids and the drops (`capacity` words each).
 */
export function parseSummitList(
  bytes: ArrayBuffer,
  capacity: number,
  width: number,
  elevation: ArrayLike<number>
): SummitList {
  const header = toUint32(bytes, 4);
  const count = Math.min(header[0], capacity);
  const ids = toUint32(bytes.slice(16), capacity);
  const drops = toFloat32(bytes.slice(16 + capacity * 4), capacity);
  const candidates: SummitCandidate[] = [];
  for (let index = 0; index < count; index++) {
    const id = ids[index];
    candidates.push({
      column: id % width,
      row: Math.floor(id / width),
      elevation: elevation[id],
      drop: drops[index]
    });
  }
  return {
    candidates,
    total: header[1],
    listOverflow: header[2] !== 0 || header[1] > capacity,
    radiusClamped: header[3] !== 0
  };
}

/** The candidates that pass a minimum drop (survivors) and the disc maxima that fail it. */
export function applyDropThreshold(
  candidates: readonly SummitCandidate[],
  minimumDrop: number
): {survivors: SummitCandidate[]; rejected: SummitCandidate[]} {
  const survivors: SummitCandidate[] = [];
  const rejected: SummitCandidate[] = [];
  for (const candidate of candidates) {
    (candidate.drop >= minimumDrop ? survivors : rejected).push(candidate);
  }
  return {survivors, rejected};
}

/** Size class of a drop: 0 for the smallest triangles up to 3 for drops over the last break. */
export function getDropClass(dropMeters: number): number {
  return SUMMIT_DROP_CLASSES.breaks.filter(limit => dropMeters >= limit).length;
}

/**
 * Counts drops in equal bins from 0; a drop beyond the last bin is counted in it.
 *
 * @param binWidth Bin width in metres.
 * @param binCount Number of bins.
 */
export function getDropHistogram(
  candidates: readonly SummitCandidate[],
  binWidth: number,
  binCount: number
): number[] {
  const bins = new Array<number>(binCount).fill(0);
  for (const {drop} of candidates) {
    if (!Number.isFinite(drop)) continue;
    bins[Math.min(binCount - 1, Math.max(0, Math.floor(drop / binWidth)))]++;
  }
  return bins;
}

// ---------------------------------------------------------------------------------------------
// The OpenStreetMap catalogue
// ---------------------------------------------------------------------------------------------

/** A named OpenStreetMap peak inside the tile, on the analysis grid. */
export type CataloguePeak = {
  /** The label as the story spells it. */
  name: string;
  /** Published elevation in metres, or null. */
  elevationMeters: number | null;
  lngLat: readonly [number, number];
  /** Analysis-grid position of the catalogue point, fractional centre indices. */
  column: number;
  row: number;
  osmId: number;
};

/** Spellings the story prefers to OpenStreetMap's (the chapter's gazetteer spells these as one word). */
const PREFERRED_NAMES: Record<string, string> = {
  'Ober Gabelhorn': 'Obergabelhorn',
  'Unter Gabelhorn': 'Untergabelhorn',
  'Mittler Gabelhorn': 'Mittlergabelhorn'
};

/**
 * The label of an OSM peak name: bilingual names ("Breithorn Occidentale / Westgipfel") keep the
 * mountain and the German part ("Breithorn Westgipfel"); a few split spellings are joined.
 */
export function getPeakLabel(osmName: string): string {
  const [first, second] = osmName.split(' / ');
  const name = second ? `${first.split(' ')[0]} ${second}` : osmName;
  return PREFERRED_NAMES[name] ?? name;
}

/**
 * The named peaks of `alps-context` that lie inside the analysis grid, with their position on it.
 * They are the real OpenStreetMap nodes, so their positions are as sloppy as the data is.
 */
export function buildCatalogue(context: AlpsContext, terrain: AlpsTerrain): CataloguePeak[] {
  const peaks: CataloguePeak[] = [];
  for (const peak of context.peaks) {
    if (!peak.name) continue;
    const [column, row] = terrain.getPixel(peak.lngLat[0], peak.lngLat[1]);
    if (column < 1 || row < 1 || column > terrain.width - 2 || row > terrain.height - 2) continue;
    peaks.push({
      name: getPeakLabel(peak.name),
      elevationMeters: peak.elevationMeters,
      lngLat: peak.lngLat,
      column,
      row,
      osmId: peak.osmId
    });
  }
  return peaks.sort((a, b) => (b.elevationMeters ?? 0) - (a.elevationMeters ?? 0));
}

/**
 * Gives each catalogue peak to the nearest survivor within {@link NAME_MATCH_METERS}. A survivor
 * keeps the nearer of two peaks. Returns the survivor index to peak.
 */
export function matchNamedSurvivors(
  survivors: readonly SummitCandidate[],
  catalogue: readonly CataloguePeak[],
  cellMeters: number
): Map<number, CataloguePeak> {
  const named = new Map<number, {peak: CataloguePeak; distance: number}>();
  const limit = NAME_MATCH_METERS / cellMeters;
  for (const peak of catalogue) {
    let best = -1;
    let bestDistance = limit;
    for (let index = 0; index < survivors.length; index++) {
      const distance = Math.hypot(
        survivors[index].column - peak.column,
        survivors[index].row - peak.row
      );
      if (distance <= bestDistance) {
        best = index;
        bestDistance = distance;
      }
    }
    if (best < 0) continue;
    const existing = named.get(best);
    if (!existing || bestDistance < existing.distance) {
      named.set(best, {peak, distance: bestDistance});
    }
  }
  return new Map([...named].map(([index, {peak}]) => [index, peak]));
}

// ---------------------------------------------------------------------------------------------
// Peak snap
// ---------------------------------------------------------------------------------------------

/** What a snap run did to every catalogue point. */
export type SnapSummary = {
  counts: Record<SnapStatusId, number>;
  /** Median ground move of the snapped points, metres; null when none moved. */
  medianMoveMeters: number | null;
};

/** Counts the status codes and finds the median move of the snapped points. */
export function summarizeSnap(
  statusCodes: ArrayLike<number>,
  distances: ArrayLike<number>,
  count: number
): SnapSummary {
  const counts = Object.fromEntries(SNAP_STATUS_IDS.map(id => [id, 0])) as Record<
    SnapStatusId,
    number
  >;
  const moves: number[] = [];
  for (let index = 0; index < count; index++) {
    const id = getSnapStatusId(statusCodes[index]);
    if (!id) continue;
    counts[id]++;
    if (id === 'snapped') moves.push(distances[index]);
  }
  moves.sort((a, b) => a - b);
  const middle = Math.floor(moves.length / 2);
  const medianMoveMeters = moves.length
    ? moves.length % 2
      ? moves[middle]
      : (moves[middle - 1] + moves[middle]) / 2
    : null;
  return {counts, medianMoveMeters};
}
