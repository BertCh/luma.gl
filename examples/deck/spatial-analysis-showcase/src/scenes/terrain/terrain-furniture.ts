// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Map furniture shared by the terrain chapter: the source credit, the camera frames (one
 * vocabulary for all seven stories), the honesty chips, the standing DEM sample line and the title
 * cartouche helper. Camera frames are pitch 0 and bearing 0 everywhere (no displaced terrain this
 * round, rule 11).
 */

import {CREDITS, joinCredits} from '../../cartography/credits';
import type {CartoucheSpec, FurnitureSpec} from '../../cartography/types';
import type {ViewState} from '../../engine/types';

/**
 * The source credit of the chapter: the relief (swisstopo swissALTI3D via Mapterhorn), the Italian
 * side of the wide DEM (regional DTMs, CC BY 4.0), Copernicus GLO-30 and OpenStreetMap (glaciers,
 * peaks, places; ODbL). Wording follows the `alps-dem-wide` attribution.
 */
export const TERRAIN_CREDIT = joinCredits(
  'Relief: © swisstopo swissALTI3D, © Mapterhorn',
  "Regione Autonoma Valle d'Aosta and Regione Piemonte DTM (CC BY 4.0)",
  CREDITS.copernicus,
  CREDITS.openStreetMap
);

/**
 * The camera frames of the chapter. `home` holds the Matterhorn, Zermatt and the Gornergrat;
 * `matterhorn` is the north face; `valley` is the Mattertal around Zermatt; `gornergratWide` is the
 * whole cirque from the station out to Monte Rosa (the wide DEM).
 */
export const TERRAIN_FRAMES = {
  home: {longitude: 7.742, latitude: 45.985, zoom: 11.9, pitch: 0, bearing: 0},
  matterhorn: {longitude: 7.676, latitude: 45.98, zoom: 13.4, pitch: 0, bearing: 0},
  valley: {longitude: 7.745, latitude: 45.995, zoom: 12.4, pitch: 0, bearing: 0},
  gornergratWide: {longitude: 7.775, latitude: 45.985, zoom: 11.2, pitch: 0, bearing: 0}
} as const satisfies Record<string, ViewState>;

/** The honesty chip of the sight-and-light stories: the DEM has no trees or buildings. */
export const BARE_EARTH_CHIP = 'Bare earth: no trees or buildings';

/** The chip of the sun story: clear sky and bare earth. */
export const CLEAR_SKY_CHIP = 'Clear sky, bare earth';

/** What a DEM needs to say about itself in the sample line. */
export type DemSampleSource = {
  width: number;
  height: number;
  /** Ground metres per cell at the central row. */
  groundCellSize: number;
};

/**
 * The standing sample line of the cartouche, computed from the DEM it describes:
 * `swissALTI3D via Mapterhorn · 6.6 m cells · 2,048 × 2,048`. An `AlpsGrid` or `AlpsTerrain` fits.
 */
export function demSampleLine(dem: DemSampleSource, source = 'swissALTI3D via Mapterhorn'): string {
  const cells = dem.groundCellSize.toFixed(1);
  const format = (value: number) => value.toLocaleString('en-US');
  return `${source} · ${cells} m cells · ${format(dem.width)} × ${format(dem.height)}`;
}

/**
 * The title cartouche of a step: line 1 the question (nine words or fewer), line 2 the variable,
 * unit, method and cell size, and the standing sample line.
 *
 * @example
 * terrainCartouche('How steep is the Matterhorn?', 'Slope, degrees · Horn 3 x 3 · 6.6 m cells', demSampleLine(grid));
 */
export function terrainCartouche(
  title: string,
  subtitle: string,
  sample?: string,
  chips?: readonly string[]
): CartoucheSpec {
  return {
    title,
    subtitle,
    ...(sample ? {sample} : {}),
    ...(chips?.length ? {chips} : {})
  };
}

/**
 * The furniture of a terrain step: the cartouche, a metric scale bar (ticked at the metric
 * parameter of the step) and the chapter credit.
 */
export function getTerrainFurniture(options: {
  cartouche: CartoucheSpec;
  /** Distances in metres marked on the scale bar (radius, window, maximum distance). */
  scaleTicks?: readonly number[];
}): FurnitureSpec {
  return {
    title: options.cartouche,
    scaleBar: options.scaleTicks?.length
      ? {units: 'metric', ticks: options.scaleTicks}
      : {units: 'metric'},
    credit: TERRAIN_CREDIT
  };
}
