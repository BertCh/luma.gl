// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The visual language of `points/street-density`: one fixed set of class breaks for the whole
 * story, the YlOrBr class tables on both grounds, the street hierarchy (how a line is drawn on the
 * night ground and on paper), the "no data" mask colours and the corridor of the mile-grid step.
 * Pure TypeScript with no luma.gl or engine imports, so the scene file can import it.
 */

import {
  DARK_GROUND_MINIMUM_LIGHTNESS,
  getClassPaletteHex,
  hexToRgba,
  liftLightnessFloor,
  makeClassTable
} from '../../cartography/class-table';
import {CHICAGO} from '../../cartography/gazetteer';
import {MAP_INK} from '../../cartography/hue-registry';
import type {ClassTable} from '../../cartography/types';
import type {ZoomStops} from '../../cartography/zoom';
import {pointsCartouche} from './b1-points-look';

/** An RGBA colour, 0-255. */
export type StreetColor = readonly [number, number, number, number];

/** A ground: what `ctx.ground()` returns. */
export type StreetGround = 'light' | 'dark';

/**
 * Fixed class breaks of street density in km of street per km² (rule 7: one classification for
 * the whole story). Classes: under 5, 5-10, 10-15, 15-20, 20 and over. A cell with no street at
 * all is transparent, not a class.
 */
export const DENSITY_BREAKS: readonly number[] = [5, 10, 15, 20];

/**
 * Fixed class breaks of the "wrong unit" view, metres of street per cell. They are fixed in
 * metres, so a bigger cell moves into higher classes on its own: the colours drift with the cell
 * size, which is the point of that view.
 */
export const LENGTH_BREAKS: readonly number[] = [500, 1000, 2000, 4000];

/**
 * A city cell is a "void" for the gap notes when its density is under this value in km of street
 * per km² (half of the lowest class break).
 */
export const VOID_DENSITY = DENSITY_BREAKS[0] / 2;

/** Tracts with fewer residents than this are set aside in the per-resident map (small numbers). */
export const SUPPRESSION_RESIDENTS = 200;

/** The cell sizes of the scale sweep (metres): the MAUP curve of step 4. */
export const CELL_SWEEP_SIZES: readonly number[] = [60, 100, 150, 250, 350, 600];

/** Bearing bins of the rose (36 bins of 10 degrees, folded so a bearing and its opposite agree). */
export const ROSE_BIN_COUNT = 36;

/** A street within this many degrees of an axis counts as "on the grid". */
export const GRID_TOLERANCE_DEGREES = 5;

/** The Public Land Survey section line spacing in metres (one statute mile). */
export const MILE_METERS = 1609.344;

/** Bin height of the arterial profile in metres. */
export const MILE_PROFILE_BIN_METERS = 100;

/** An arterial centreline within this many metres of a fitted lattice line counts as "on" it. */
export const LATTICE_TOLERANCE_METERS = 100;

/**
 * The corridor of the mile-grid step in degrees `[west, south, east, north]`, from gazetteer
 * places: west of the Loop, from south of Midway to north of Garfield Park, which holds the
 * one-mile South Side and the half-mile North Side arterials.
 */
export const MILE_CORRIDOR: readonly [number, number, number, number] = (() => {
  const midway = CHICAGO.places.midway.lngLat;
  const garfield = CHICAGO.places['garfield-park'].lngLat;
  const loop = CHICAGO.places.loop.lngLat;
  return [midway[0] - 0.07, midway[1] - 0.046, loop[0], garfield[1] + 0.067];
})();

/**
 * The camera of the mile-grid step: the middle of Garfield Park, Bridgeport and Midway at about
 * z11.2, so the West and South Side lattices fill the frame.
 */
export const MILE_VIEW = (() => {
  const places = ['garfield-park', 'bridgeport', 'midway'].map(id => CHICAGO.places[id].lngLat);
  const longitudes = places.map(place => place[0]);
  const latitudes = places.map(place => place[1]);
  return {
    longitude: (Math.min(...longitudes) + Math.max(...longitudes)) / 2,
    latitude: (Math.min(...latitudes) + Math.max(...latitudes)) / 2,
    zoom: 11.2
  } as const;
})();

// ---------------------------------------------------------------------------------------------
// Class tables
// ---------------------------------------------------------------------------------------------

/**
 * The five class colours on a ground: the YlOrBr 6-class table without its near-white lowest
 * class (zero is transparent, so that class has no job). On the dark ground the authored dark
 * table is lifted so its lowest class keeps 3:1 against the ground.
 */
export function getDensityColors(ground: StreetGround): StreetColor[] {
  const hexes = getClassPaletteHex('YlOrBr', 6, {ground}).slice(1);
  const lifted =
    ground === 'dark' ? liftLightnessFloor(hexes, DARK_GROUND_MINIMUM_LIGHTNESS) : hexes;
  return lifted.map(hex => hexToRgba(hex, 255) as StreetColor);
}

/** The fixed class table of the density field (`mode: 'density'`) or the wrong-unit view. */
export function makeDensityTable(
  ground: StreetGround,
  mode: 'density' | 'length' = 'density'
): ClassTable {
  return makeClassTable({
    breaks: mode === 'density' ? DENSITY_BREAKS : LENGTH_BREAKS,
    colors: getDensityColors(ground),
    unit: mode === 'density' ? 'km of street per km²' : 'm of street per cell',
    method: 'Fixed breaks',
    noData: {label: 'Outside the city: no data'}
  });
}

// ---------------------------------------------------------------------------------------------
// Streets
// ---------------------------------------------------------------------------------------------

/** One line style of the street layers. */
export type StreetLineStyle = {widthPixels: ZoomStops; color: StreetColor};

/** Styles of the two street tiers. */
export type StreetStyle = {local: StreetLineStyle; arterial: StreetLineStyle};

/** Warm white of the streets on the night ground. */
const NIGHT_STREET_RGB = [255, 244, 224] as const;

/**
 * How the streets are drawn. Night, `all`: every street a thin warm-white line, motorways and
 * arterials heavier. Night, `arterials`: arterials bright, local streets pushed back to context.
 * Over a classed field (`under`): hairlines in the context grey, under the field.
 */
export function getStreetStyle(
  ground: StreetGround,
  emphasis: 'all' | 'arterials',
  under: boolean
): StreetStyle {
  if (under) {
    const context = hexToRgba(MAP_INK[ground].context);
    const alpha = Math.round(0.45 * 255);
    return {
      local: {
        widthPixels: [
          [10, 0.3],
          [13, 0.5],
          [16, 0.8]
        ],
        color: [context[0], context[1], context[2], alpha]
      },
      arterial: {
        widthPixels: [
          [10, 0.5],
          [13, 0.8],
          [16, 1.2]
        ],
        color: [context[0], context[1], context[2], alpha]
      }
    };
  }
  const [red, green, blue] = ground === 'dark' ? NIGHT_STREET_RGB : ([31, 41, 51] as const);
  if (emphasis === 'arterials') {
    return {
      local: {
        widthPixels: [
          [9, 0.3],
          [11, 0.4],
          [14, 0.8]
        ],
        color: [red, green, blue, 64]
      },
      arterial: {
        widthPixels: [
          [9, 0.9],
          [11, 1.2],
          [14, 2]
        ],
        color: [red, green, blue, 255]
      }
    };
  }
  return {
    local: {
      widthPixels: [
        [9, 0.5],
        [10.25, 0.6],
        [13, 1],
        [16, 1.5]
      ],
      color: [red, green, blue, 150]
    },
    arterial: {
      widthPixels: [
        [9, 0.9],
        [10.25, 1.1],
        [13, 1.6],
        [16, 2.4]
      ],
      color: [red, green, blue, 225]
    }
  };
}

/** Swatch colours of the street legend (the layers' own colours at the step's emphasis). */
export function getStreetLegendEntries(
  ground: StreetGround,
  emphasis: 'all' | 'arterials'
): {color: StreetColor; widthPixels: number; label: string}[] {
  const style = getStreetStyle(ground, emphasis, false);
  return [
    {
      color: style.arterial.color,
      widthPixels: emphasis === 'arterials' ? 1.2 : 1.1,
      label: 'Motorways to secondary roads'
    },
    {
      color: style.local.color,
      widthPixels: emphasis === 'arterials' ? 0.5 : 0.7,
      label: 'Tertiary, residential and service streets'
    }
  ];
}

// ---------------------------------------------------------------------------------------------
// The edge: outside the city is no data
// ---------------------------------------------------------------------------------------------

/**
 * Fill of the "outside the city" mask above the field: the night ground colour at 0.85, the paper
 * land colour at 0.9. The lake keeps its water colour on paper so the shore still reads.
 */
export function getOutsideColor(ground: StreetGround): StreetColor {
  return ground === 'dark' ? [11, 14, 18, 217] : [242, 240, 234, 230];
}

/** Lake fill drawn over the mask on paper (the paper ground's water colour). */
export const LAKE_PAPER_COLOR: StreetColor = [214, 224, 230, 230];

/** Fill opacity of the density field: lower in `both` so the lines under it show through. */
export function getFieldOpacity(ground: StreetGround, layers: 'field' | 'both'): number {
  if (layers === 'both') return ground === 'dark' ? 0.7 : 0.6;
  return ground === 'dark' ? 0.9 : 0.82;
}

// ---------------------------------------------------------------------------------------------
// Furniture and wording
// ---------------------------------------------------------------------------------------------

/** The cartouche of a step: the claim and the variable line, without the observer-effort chip. */
export function streetCartouche(claim: string, variable: string) {
  return pointsCartouche(claim, variable, {effort: false});
}

/** The variable line of the field steps. */
export const FIELD_VARIABLE = 'km of street per km², 350 m cells, OpenStreetMap';

/** The lines every classed value is drawn with: one white hairline on paper, dark on the dark ground. */
export function getPolygonLineColor(ground: StreetGround): StreetColor {
  return ground === 'dark' ? [14, 17, 22, 128] : [255, 255, 255, 161];
}
