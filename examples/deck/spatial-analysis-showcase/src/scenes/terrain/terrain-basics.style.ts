// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Pure (luma-free) pieces of the terrain-basics scene shared by the scene file (loaded by the
 * gallery) and its compute module: the option state, the class tables of every product (one
 * object that the layer, the legend, the tooltip and the histogram all read), the legends and the
 * live subtitle of the title cartouche.
 */

import {getClassTableLegend, hexToRgba} from '../../cartography/class-table';
import {MAP_INK} from '../../cartography/hue-registry';
import type {ClassTable} from '../../cartography/types';
import type {LegendSpec} from '../scene';
import {
  ASPECT_SLOPE_ALPHA,
  getAspectLegend,
  getSignedLegend,
  makeElevationTable,
  makeRuggednessTable,
  makeSignedClassTable,
  makeSlopeTable,
  SLOPE_BREAKS,
  SLOPE_CONTINUOUS,
  TPI_BREAKS_METERS,
  type TerrainGroundTone
} from './terrain-palettes';

type GPUTerrainRGBEncoding = 'terrarium' | 'mapbox';
type GPUTerrainRuggednessAlgorithm = 'riley' | 'wilson';
type GPUTerrainRuggednessEdgeMode = 'nodata' | 'extrapolate';

/** What the map shows: the relief ground, the GPU-decoded heights, or an analysis product. */
export type BasicsView = 'relief' | 'decoded' | 'analysis';

/** The analysis products of the scene. */
export type BasicsProduct = 'slope' | 'aspect' | 'tpi' | 'tri' | 'vrm';

/** Option state of the terrain-basics scene. */
export type BasicsOptions = {
  view: BasicsView;
  product: BasicsProduct;
  encoding: GPUTerrainRGBEncoding;
  validRange: 'default' | 'tight' | 'off';
  alphaNoData: boolean;
  clampBathymetry: boolean;
  missingTile: boolean;
  spikeDensity: number;
  repairSpikes: boolean;
  slopeDisplay: 'classes' | 'continuous';
  cellModel: 'ground' | 'mercator';
  zFactor: number;
  borderMode: 'clamp' | 'nodata';
  triAlgorithm: GPUTerrainRuggednessAlgorithm;
  edgeMode: GPUTerrainRuggednessEdgeMode;
  vrmRadius: number;
};

/** Valid height ranges of the decode, metres. */
export const VALID_RANGES: Record<BasicsOptions['validRange'], readonly [number, number]> = {
  default: [-11000, 9000],
  tight: [1800, 4200],
  off: [-Infinity, Infinity]
};

/**
 * Ground metres per cell of the alps-dem tile at its central row (`cellSizeGroundM` of its
 * manifest). Only used for the slider description; the readouts come from the loaded grid.
 */
export const GROUND_CELL_METERS = 6.64;

/** The decision threshold of avalanche practice: the second slope class break. */
export const DECISION_SLOPE_DEGREES = SLOPE_BREAKS[1];

/** The steep-face threshold of the second readout: the fifth slope class break. */
export const STEEP_FACE_DEGREES = SLOPE_BREAKS[4];

/** Alpha of the slope and aspect colours over the relief ground, 0-1. */
export const PRODUCT_ALPHA = 0.72;

/** Aspect colours are drawn at this alpha on slopes steeper than the full-colour threshold. */
export const ASPECT_ALPHA = 0.8;

/** Tint strength of the decoded-height classes of step 2 (the relief ground uses 0.3). */
const DECODED_TINT_STRENGTH = 0.85;

/** Quantile probabilities of the ruggedness classes: median, 75th, 90th and 98th percentile. */
export const RUGGEDNESS_PROBABILITIES = [0.5, 0.75, 0.9, 0.98] as const;

/** The class tables of every product on one ground, as the layer and the legends read them. */
export type BasicsTables = {
  /** The elevation tint as it reads on the relief ground (legend of step 1). */
  elevation: ClassTable;
  /** The same classes at a stronger tint, drawn opaque in the decode step. */
  decoded: ClassTable;
  slope: ClassTable;
  tpi: ClassTable;
  /** TRI or VRM classes at frozen quantile breaks. */
  rugged: ClassTable;
};

/** Frozen ruggedness classes of one product and configuration. */
export type RuggednessBreaks = {product: 'tri' | 'vrm'; breaks: readonly number[]};

/** The hatch colour of no-data cells, RGBA 0-255. */
export function getHatchColor(ground: TerrainGroundTone): [number, number, number, number] {
  const [red, green, blue] = hexToRgba(MAP_INK[ground].inkMuted);
  return [red, green, blue, 200];
}

/**
 * Builds every class table for a ground. `ruggedness` holds the quantile breaks of the current
 * TRI or VRM product (null before the tile has been measured: the classes are then empty).
 */
export function makeBasicsTables(
  ground: TerrainGroundTone,
  ruggedness: RuggednessBreaks | null
): BasicsTables {
  const decoded = makeElevationTable(ground, DECODED_TINT_STRENGTH);
  const product = ruggedness?.product ?? 'tri';
  return {
    elevation: makeElevationTable(ground),
    decoded: {
      ...decoded,
      method: 'Alpine tint classes of the GPU-decoded heights',
      noData: {
        color: getHatchColor(ground),
        label: 'No data (alpha 0 or out of range)',
        hatched: true
      }
    },
    slope: makeSlopeTable(ground, Math.round(PRODUCT_ALPHA * 255)),
    tpi: makeSignedClassTable(TPI_BREAKS_METERS, ground, 'm', {
      method: 'Fixed classes in metres; the middle class is not drawn',
      format: value => `${value} m`
    }),
    rugged: makeRuggednessTable(
      ruggedness?.breaks ?? [0, 0, 0, 0],
      ground,
      product === 'tri' ? 'm' : 'index 0-1',
      {
        format: value => (product === 'tri' ? value.toFixed(1) : value.toFixed(3))
      }
    )
  };
}

/** True once the quantile breaks of a ruggedness product have been measured (not all zero). */
function hasBreaks(breaks: readonly number[] | undefined): breaks is readonly number[] {
  return Boolean(breaks?.some(value => value > 0));
}

/** Window width in metres of a `(2r + 1)` cell window. */
export function getWindowMeters(radiusCells: number, cellMeters: number): number {
  return (2 * radiusCells + 1) * cellMeters;
}

/** Subtitle (line 2 of the cartouche): variable, unit, method and cell size, live from the options. */
export function getCartoucheSubtitle(
  state: BasicsOptions,
  cellMeters: number,
  mercatorMeters: number
): string {
  const cell = cellMeters.toFixed(1);
  if (state.view === 'relief') {
    return `Elevation, metres · pale tint under hillshade · ${cell} m cells`;
  }
  if (state.view === 'decoded') {
    return `Heights decoded on the GPU as ${
      state.encoding === 'terrarium' ? 'Terrarium' : 'Mapbox'
    } · ${cell} m cells`;
  }
  switch (state.product) {
    case 'slope': {
      const window = Math.round(3 * (state.cellModel === 'ground' ? cellMeters : mercatorMeters));
      return state.cellModel === 'ground'
        ? `Slope, degrees · Horn 3 × 3 = ${window} m window · ${cell} m ground cells`
        : `Slope, degrees · Mercator pixels as metres (${mercatorMeters.toFixed(1)} m): wrong`;
    }
    case 'aspect':
      return `Aspect, direction faced downhill · coloured only on slopes over ${ASPECT_SLOPE_ALPHA.flatDegrees}°`;
    case 'tpi':
      return `TPI, metres above or below the 8 neighbours · 3 × 3 = ${Math.round(3 * cellMeters)} m window`;
    case 'tri':
      return `TRI, metres · Riley 3 × 3 = ${Math.round(3 * cellMeters)} m window · quantile classes`;
    case 'vrm': {
      const size = 2 * state.vrmRadius + 1;
      return `VRM, index 0 to 1 · ${size} × ${size} = ${Math.round(getWindowMeters(state.vrmRadius, cellMeters))} m window`;
    }
  }
}

/** The legends of the current state; tables come from `data.tables` (the ground the scene is on). */
export function getBasicsLegends(
  state: BasicsOptions,
  data: Readonly<Record<string, unknown>>
): LegendSpec[] {
  const tables = (data['tables'] as BasicsTables | undefined) ?? makeBasicsTables('light', null);
  if (state.view === 'relief') {
    return [
      getClassTableLegend(tables.elevation, {
        title: 'Elevation (m above sea level)',
        layout: 'bar',
        id: 'elevation',
        note: 'Pale tint under the shaded relief; glaciers outlined by OpenStreetMap'
      })
    ];
  }
  if (state.view === 'decoded') {
    return [
      getClassTableLegend(tables.decoded, {
        title: 'Elevation decoded on the GPU (m)',
        layout: 'bar',
        id: 'decoded'
      })
    ];
  }
  switch (state.product) {
    case 'slope': {
      if (state.slopeDisplay === 'continuous') {
        return [
          {
            kind: 'ramp',
            id: 'slope-continuous',
            title: 'Slope, continuous colour',
            ramp: SLOPE_CONTINUOUS.ramp,
            extent: SLOPE_CONTINUOUS.domain,
            unit: '°',
            note: `Transparent below ${SLOPE_CONTINUOUS.transparentBelowDegrees}°. Nothing marks the ${DECISION_SLOPE_DEGREES}° threshold.`
          }
        ];
      }
      const counts = (
        data['slopeCounts'] as Partial<Record<'ground' | 'mercator', number[]>> | undefined
      )?.[state.cellModel];
      const histogram = data['slopeHistogram'] as number[] | undefined;
      return [
        getClassTableLegend(tables.slope, {
          title:
            state.cellModel === 'ground' ? 'Slope on ground cells' : 'Slope on Mercator pixels',
          id: 'slope',
          ...(counts ? {counts, layout: 'list' as const} : {layout: 'bar' as const}),
          ...(histogram ? {histogram} : {}),
          note:
            state.cellModel === 'ground'
              ? `Decision thresholds (avalanche practice). Cells under ${SLOPE_BREAKS[0]}° are not drawn.`
              : `Mercator pixels read as metres: every class shrinks. Cells under ${SLOPE_BREAKS[0]}° are not drawn.`
        })
      ];
    }
    case 'aspect':
      return [getAspectLegend()];
    case 'tpi':
      return [
        getSignedLegend(tables.tpi, 'Topographic position (TPI, m)', {
          neutralNote:
            'Orange: above its 8 neighbours (ridge). Purple: below (hollow). Middle: not drawn.'
        })
      ];
    case 'tri':
    case 'vrm': {
      const measured = data['ruggednessBreaks'] as RuggednessBreaks | undefined;
      const known = measured && measured.product === state.product && hasBreaks(measured.breaks);
      return [
        getClassTableLegend(tables.rugged, {
          title:
            state.product === 'tri' ? 'Terrain ruggedness (TRI, m)' : 'Vector ruggedness (VRM)',
          id: 'ruggedness',
          layout: 'bar',
          note: known
            ? 'Quantile classes of this tile, frozen for the step: median, 75th, 90th, 98th percentile. The lowest class is not drawn.'
            : 'Quantile classes are measured from the tile once the product is built.'
        })
      ];
    }
  }
}
