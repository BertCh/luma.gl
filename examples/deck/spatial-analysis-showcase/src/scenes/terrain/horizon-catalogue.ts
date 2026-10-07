// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The peak catalogue of the horizon scene: the named OpenStreetMap summits of the `alps-context`
 * dataset above a height floor, one per massif, each snapped to the highest cell of the wide DEM.
 * They are the targets of `GPUPointHorizonVisibility` and the labels of the map and the panorama.
 * Pure TypeScript: no GPU, no DOM.
 */

import type {TerrainDem} from './cpu-dem';
import {type AlpsContext, snapLngLatToHighestCell} from './terrain-places';
import {
  MINIMUM_PEAK_DISTANCE_METERS,
  MINIMUM_PEAK_ELEVATION_METERS,
  PEAK_THINNING_METERS
} from './horizon-style';

/** Largest catalogue the scene keeps (the size of its GPU buffers). */
export const MAXIMUM_CATALOGUE_SIZE = 64;

/** One catalogue summit. */
export type CataloguePeak = {
  /** OpenStreetMap id. */
  id: number;
  /** The OpenStreetMap name as it is. */
  name: string;
  /** The name as the labels spell it ("Breithorn Westgipfel", "Signalkuppe"). */
  displayName: string;
  /** Published elevation of the OpenStreetMap node, metres. */
  publishedElevationMeters: number;
  /** Elevation of the DEM cell the summit snapped to, metres. */
  demElevationMeters: number;
  /** Pixel-centre column and row of the snapped cell. */
  column: number;
  row: number;
  /** `[longitude, latitude]` of the snapped cell. */
  lngLat: readonly [number, number];
  /** Layer metres `[x, y]` of the snapped cell. */
  meters: readonly [number, number];
};

/**
 * The spelling a label uses. OpenStreetMap lists summits with several names ("Punta Gnifetti /
 * Signalkuppe", "Breithorn Occidentale / Westgipfel"): the part with a German summit suffix wins,
 * and a bare "Westgipfel" borrows the first word of the other part.
 */
export function getPeakDisplayName(name: string): string {
  const parts = name.split(/ \/ | - /).map(part => part.trim());
  if (parts.length === 1) return name;
  const german = parts.find(part => /(horn|spitze|kuppe|gipfel|stock|hubel)$/i.test(part));
  if (!german) return parts[0];
  return /^(west|ost|nord|süd|sued)gipfel$/i.test(german)
    ? `${parts[0].split(' ')[0]} ${german}`
    : german;
}

/** Lower case, letters and digits only, so "Ober Gabelhorn" and "Obergabelhorn" compare equal. */
function normalizeName(name: string): string {
  return name
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '');
}

/**
 * The peaks a story names first, in order. A name matches a catalogue summit when the summit's
 * display name starts with it, so "Liskamm" finds "Liskamm Ostgipfel".
 */
export const LABEL_PREFERENCE: readonly string[] = [
  'Matterhorn',
  'Dufourspitze',
  'Weisshorn',
  'Dom',
  'Dent Blanche',
  'Liskamm',
  'Breithorn',
  'Castor',
  'Ober Gabelhorn',
  "Dent d'Hérens",
  'Klein Matterhorn',
  'Zinalrothorn'
];

/**
 * Catalogue indices in label order: the preferred names first, then the rest by height. The
 * Matterhorn must not match "Klein Matterhorn", so a preference matches the whole name or a
 * following word of it ("Liskamm" matches "Liskamm Ostgipfel").
 */
export function getLabelOrder(catalogue: readonly CataloguePeak[]): number[] {
  const chosen: number[] = [];
  for (const preferred of LABEL_PREFERENCE) {
    const key = normalizeName(preferred);
    const index = catalogue.findIndex(
      (peak, candidate) =>
        !chosen.includes(candidate) &&
        (normalizeName(peak.displayName) === key ||
          peak.displayName.toLowerCase().startsWith(`${preferred.toLowerCase()} `))
    );
    if (index >= 0) chosen.push(index);
  }
  const rest = catalogue
    .map((_, index) => index)
    .filter(index => !chosen.includes(index))
    .sort(
      (left, right) =>
        catalogue[right].publishedElevationMeters - catalogue[left].publishedElevationMeters
    );
  return [...chosen, ...rest];
}

/**
 * Builds the catalogue: named summits at or above {@link MINIMUM_PEAK_ELEVATION_METERS}, snapped
 * to the highest DEM cell within 160 m, the highest first; a summit within
 * {@link PEAK_THINNING_METERS} of a higher one is dropped (one summit per massif), and so is one
 * within {@link MINIMUM_PEAK_DISTANCE_METERS} of the eye.
 *
 * @param getMeters Layer metres of a pixel-centre column and row.
 */
export function buildPeakCatalogue(
  context: AlpsContext,
  dem: TerrainDem,
  getMeters: (column: number, row: number) => [number, number],
  eyeMeters: readonly [number, number]
): CataloguePeak[] {
  const margin = 3;
  const candidates = context.peaks
    .filter(
      peak =>
        peak.name &&
        peak.elevationMeters !== null &&
        peak.elevationMeters >= MINIMUM_PEAK_ELEVATION_METERS
    )
    .sort((left, right) => (right.elevationMeters ?? 0) - (left.elevationMeters ?? 0));
  const catalogue: CataloguePeak[] = [];
  for (const peak of candidates) {
    const snapped = snapLngLatToHighestCell(dem, peak.lngLat);
    if (!snapped) continue;
    const {column, row} = snapped;
    if (
      column < margin ||
      row < margin ||
      column >= dem.width - margin ||
      row >= dem.height - margin
    ) {
      continue;
    }
    const meters = getMeters(column, row);
    if (
      Math.hypot(meters[0] - eyeMeters[0], meters[1] - eyeMeters[1]) < MINIMUM_PEAK_DISTANCE_METERS
    ) {
      continue;
    }
    const crowded = catalogue.some(
      kept =>
        Math.hypot(kept.meters[0] - meters[0], kept.meters[1] - meters[1]) < PEAK_THINNING_METERS
    );
    if (crowded) continue;
    catalogue.push({
      id: peak.osmId,
      name: peak.name as string,
      displayName: getPeakDisplayName(peak.name as string),
      publishedElevationMeters: peak.elevationMeters as number,
      demElevationMeters: snapped.elevation,
      column,
      row,
      lngLat: snapped.lngLat,
      meters
    });
    if (catalogue.length >= MAXIMUM_CATALOGUE_SIZE) break;
  }
  return catalogue;
}
