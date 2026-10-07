// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Pure (luma-free) pieces of the horizon scene shared by the scene file (loaded by the gallery)
 * and its compute module: the option state, the constants of the peak catalogue and the
 * panorama, the skyline class table, the legends and the live subtitle of the title cartouche.
 */

import {getClassTableLegend} from '../../cartography/class-table';
import {hexToRgba, MAP_INK} from '../../cartography/hue-registry';
import type {ClassTable} from '../../cartography/types';
import type {LegendSpec} from '../scene';
import {
  getVisibilitySymbolLegend,
  makeSkylineTable,
  type TerrainGroundTone
} from './terrain-palettes';

/**
 * What the map shows. Each view adds one layer of the computation: `plain` is the unanalysed
 * establishing shot, `rays` the ray fan and the skyline ring, `skyline` the ring alone,
 * `skyline-peaks` the maxima of the skyline, `visibility` the classified peaks and one cast ray,
 * `sky-view` the horizon computed for every cell.
 */
export type HorizonView =
  | 'plain'
  | 'rays'
  | 'skyline'
  | 'skyline-peaks'
  | 'visibility'
  | 'sky-view';

/** Option state of the horizon scene. */
export type HorizonOptions = {
  view: HorizonView;
  observerHeight: number;
  maxDistance: number;
  bearing: number;
  azimuthCount: '180' | '720' | '1440';
  minProminence: number;
  toleranceDegrees: number;
  gridRadius: '64' | '128' | '256';
  gridDirections: '8' | '16' | '32';
  refraction: 'none' | 'mt-image' | 'gdal';
  projection: 'web-mercator' | 'planar';
  traversal: 'pyramid' | 'march';
  peakWindow: '8' | '16' | '32' | '64';
  sigmaZ: number;
  skylineToleranceDegrees: number;
  targetIgnoreDistance: number;
  targetIgnoreFraction: number;
  gridAlgorithm: 'march' | 'sweep';
};

/** Named OpenStreetMap summits below this published elevation are not in the catalogue, metres. */
export const MINIMUM_PEAK_ELEVATION_METERS = 3500;

/** Of two catalogue summits closer than this, the lower is dropped (one per massif), metres. */
export const PEAK_THINNING_METERS = 1500;

/** Summits closer to the eye than this are not catalogued (the terrace itself), metres. */
export const MINIMUM_PEAK_DISTANCE_METERS = 600;

/** Peaks farther than this are drawn in a muted ink: the aerial-perspective cue of the labels. */
export const FAR_PEAK_METERS = 10000;

/**
 * Bearing the view opens on, degrees clockwise from north: west-south-west, where the Matterhorn
 * stands with the Dent d'Herens behind it. It was read off the wide DEM (Matterhorn 266, Dent
 * d'Herens 264 from the Gornergrat cell); the readouts report what the GPU finds.
 */
export const DEFAULT_BEARING_DEGREES = 264;

/** The panorama starts at south, so the Matterhorn, Weisshorn and Dom sit in its left half. */
export const PANORAMA_START_DEGREES = 180;

/** Skyline angle bins: below the horizontal, then 3 degrees wide. */
export const SKYLINE_FIRST_BREAK_DEGREES = 0;

/** Sky-view factor range of the grey multiply layer and its legend. */
export const SKY_VIEW_RANGE = [0.5, 1] as const;

/** Share of the grey ramp used by the sky-view layer (the darkest cell is not black). */
export const SKY_VIEW_RAMP_RANGE = [0.4, 1] as const;

const COMPASS = [
  'N',
  'NNE',
  'NE',
  'ENE',
  'E',
  'ESE',
  'SE',
  'SSE',
  'S',
  'SSW',
  'SW',
  'WSW',
  'W',
  'WNW',
  'NW',
  'NNW'
];

/** Compass point of a bearing in degrees clockwise from north. */
export function getCompassLabel(bearingDegrees: number): string {
  return COMPASS[Math.round((((bearingDegrees % 360) + 360) % 360) / 22.5) % 16];
}

/** `264° WSW`: a bearing, wrapped into 0 to 360, with its compass point. */
export function formatBearing(bearingDegrees: number): string {
  const wrapped = ((Math.round(bearingDegrees) % 360) + 360) % 360;
  return `${wrapped}° ${getCompassLabel(wrapped)}`;
}

/** The skyline class table of a ground (3-degree bins, the first class is below the horizontal). */
export function makeHorizonSkylineTable(ground: TerrainGroundTone): ClassTable {
  return makeSkylineTable(ground, {
    binDegrees: 3,
    firstBreakDegrees: SKYLINE_FIRST_BREAK_DEGREES
  });
}

/** Subtitle (line 2 of the cartouche): variable, unit, method and the live options. */
export function getHorizonSubtitle(
  state: HorizonOptions,
  info: {rayCount: number; gridRadiusKilometers: number}
): string {
  const eye = `eye ${state.observerHeight} m`;
  switch (state.view) {
    case 'plain':
      return `Named summits above ${MINIMUM_PEAK_ELEVATION_METERS.toLocaleString('en-US')} m · ${eye}`;
    case 'rays':
    case 'skyline':
      return `Elevation angle of the skyline · ${info.rayCount} rays · ${eye}`;
    case 'skyline-peaks':
      return `Skyline maxima, prominence in degrees · ${info.rayCount} rays · ${eye}`;
    case 'visibility':
      return `Peak angle against the highest ground in front · ${eye}`;
    case 'sky-view':
      return `Sky-view factor · ${state.gridDirections} sectors · ${info.gridRadiusKilometers.toFixed(1)} km radius`;
  }
}

/** The tables the layers and the legends read, for one ground. */
export type HorizonTables = {
  ground: TerrainGroundTone;
  skyline: ClassTable;
};

/** Builds the tables of a ground. */
export function makeHorizonTables(ground: TerrainGroundTone): HorizonTables {
  return {ground, skyline: makeHorizonSkylineTable(ground)};
}

/** The legends of the current view; tables come from `data.tables` (the ground the scene is on). */
export function getHorizonLegends(
  state: HorizonOptions,
  data: Readonly<Record<string, unknown>>
): LegendSpec[] {
  const tables = (data['tables'] as HorizonTables | undefined) ?? makeHorizonTables('light');
  switch (state.view) {
    case 'plain':
      return [];
    case 'rays':
    case 'skyline':
    case 'skyline-peaks':
      return [
        getClassTableLegend(tables.skyline, {
          title: 'Skyline elevation angle',
          id: 'skyline',
          layout: 'bar',
          note: 'Degrees above the horizontal, 3-degree bins; the first class is below the eye'
        })
      ];
    case 'visibility': {
      const symbols = getVisibilitySymbolLegend(tables.ground);
      if (symbols.kind !== 'categories') return [symbols];
      return [
        {
          ...symbols,
          entries: [
            ...symbols.entries,
            {
              color: hexToRgba(MAP_INK[tables.ground].ink),
              label: 'Ink ring: on the skyline',
              shape: 'ring'
            }
          ],
          note: 'A peak is visible when its elevation angle clears the highest ground in front of it'
        }
      ];
    }
    case 'sky-view':
      return [
        {
          kind: 'ramp',
          id: 'sky-view',
          title: 'Sky-view factor',
          ramp: 'grayscale',
          extent: SKY_VIEW_RANGE,
          range: SKY_VIEW_RAMP_RANGE,
          unit: 'share of the sky seen',
          labels: ['0.5 enclosed', '1 open sky'],
          note: 'Multiplied into the relief: darker cells see less sky'
        }
      ];
  }
}
