// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The visual language of the flight-bundling story, in one place (no GPU imports, so the scene
 * file and the legends can use it).
 *
 * - **Corridors are cool, volume is gold.** The chapter's gold (`FLOW_INK`) means volume. Bundles
 *   are corridors of grouped routes, not volume, so they get the cool `mako` ramp, trimmed to
 *   `[0.3, 1]` so the shortest class stays visible on the night ground, and the two meanings are
 *   never confused. Short routes are dim, long routes bright (luminance carries the hierarchy).
 * - **Delay is an intensity.** The US step uses magma trimmed to t 0.45-1 (harm on a dark ground).
 * - One slot table feeds the layer palette, the legend and the tooltips.
 */

import {makeClassTable, hexToRgba} from '../../cartography/class-table';
import {OKABE_ITO_DARK_HEXES} from '../../cartography/hue-registry';
import type {ClassTable} from '../../cartography/types';
import {getClassColors, type PaletteColor} from '../../engine/ramps';
import {CONTEXT_INK} from './flows-style';

/** Interior breaks of the distance classes, in km: under 500, 500-1,500, 1,500-3,000, 3,000-6,000, over. */
export const DISTANCE_BREAKS: readonly number[] = [500, 1500, 3000, 6000];

/** Interior breaks of the mean departure delay classes, in minutes. */
export const DELAY_BREAKS: readonly number[] = [10, 20, 30, 45];

/**
 * Alpha of each distance class on the additive night ground (short dim, long bright). Additive
 * light saturates quickly where thousands of routes overlap, so these are low.
 */
export const DISTANCE_ALPHA: readonly number[] = [0.1, 0.13, 0.17, 0.22, 0.28];

/** Alpha of each delay class (normal blending, worst drawn last): the worst corridors lead. */
export const DELAY_ALPHA: readonly number[] = [0.45, 0.55, 0.65, 0.8, 0.92];

/** The single colour of the unclassed hairball and its alpha (additive). */
export const PLAIN_COLOR: PaletteColor = [143, 184, 222, 255];
export const PLAIN_ALPHA = 0.12;

/** Alpha of a pair under the flight floor (grey, "too few flights to trust"). */
export const UNDER_FLOOR_ALPHA = 0.15;

/** Alpha of the straight ghost under the bundles; drawn half a pixel wide, so it shows at about 0.10. */
export const GHOST_ALPHA = 0.2;

/** Alpha multiplier of every non-probe route while the probe routes are shown. */
export const PROBE_DIM = 0.35;

/** The white ring of hub airports: dark fill, 1.4 px white stroke, no orange. */
export const HUB_FILL: PaletteColor = [10, 13, 18, 204];
export const HUB_STROKE: PaletteColor = [255, 255, 255, 255];

/** Palette slots consumed by the bundled route ribbon layer. */
export const SLOT = {
  /** Slots 0 to 4: the five classes of the active measure. */
  firstClass: 0,
  underFloor: 5,
  plain: 6,
  ghost: 7,
  /** With probes active, slots 5 to 7 are reused by the three world-network probe routes. */
  firstProbe: 5
} as const;

/** A probe route: two airports looked up by IATA code, with fallbacks for 2014 gaps. */
export type ProbeRoute = {
  id: string;
  /** IATA code of the first airport. */
  from: string;
  /** IATA codes tried in order for the second airport (the first pair that exists wins). */
  to: readonly string[];
  /** Okabe-Ito dark index of the flat colour. */
  hueIndex: number;
};

/**
 * Three probe routes of the radius step. The 2014 OpenFlights data has no New York to Haneda
 * pair, so Narita stands in as the second Tokyo airport.
 */
export const PROBE_ROUTES: readonly ProbeRoute[] = [
  {id: 'north-atlantic', from: 'JFK', to: ['LHR'], hueIndex: 3},
  {id: 'trans-pacific', from: 'JFK', to: ['HND', 'NRT'], hueIndex: 4},
  {id: 'europe-asia', from: 'LHR', to: ['SIN'], hueIndex: 1}
];

/** Flat colour of a probe route. */
export function getProbeColor(probe: ProbeRoute): PaletteColor {
  return hexToRgba(OKABE_ITO_DARK_HEXES[probe.hueIndex]);
}

/** Corridor anchor routes: the name goes on the middle of this bundled route. */
export type CorridorAnchor = {
  id: string;
  name: string;
  from: string;
  to: readonly string[];
};

/** The corridors the bundle step names, each snapped to the bundled path of a representative route. */
export const CORRIDOR_ANCHORS: readonly CorridorAnchor[] = [
  {id: 'north-atlantic', name: 'North Atlantic', from: 'JFK', to: ['LHR']},
  {id: 'europe-gulf', name: 'Europe to the Gulf', from: 'FRA', to: ['DXB']},
  {id: 'trans-pacific', name: 'Trans-Pacific', from: 'LAX', to: ['HND', 'NRT']}
];

/** The grey of routes under the flight floor (the chapter's context ink, dark ground). */
export const UNDER_FLOOR_COLOR: PaletteColor = [
  CONTEXT_INK.dark[0],
  CONTEXT_INK.dark[1],
  CONTEXT_INK.dark[2],
  255
];

/** Opaque class colours of the distance classes (cool mako, trimmed), low class first. */
export function getDistanceColors(): PaletteColor[] {
  return getClassColors('mako', 5, false, 255, [0.3, 1]);
}

/**
 * Opaque class colours of the delay classes, low class first: magma (the harm family's dark
 * variant) trimmed to t 0.45-1 so the lowest class keeps contrast against the night ground (the
 * authored 5-class dark table starts at a purple barely above the ground).
 */
export function getDelayColors(): PaletteColor[] {
  return getClassColors('magma', 5, false, 255, [0.45, 1]);
}

/** The distance class table: the layer palette, the legend and the tooltip read this one object. */
export function getDistanceTable(): ClassTable {
  return makeClassTable({
    breaks: DISTANCE_BREAKS,
    colors: getDistanceColors(),
    unit: 'km',
    extent: [0, 15000],
    method: 'Fixed classes of great-circle distance',
    format: value => value.toLocaleString('en-US'),
    noData: {label: 'Hidden by the filters', color: UNDER_FLOOR_COLOR}
  });
}

/** The delay class table, with the grey "under the floor" swatch as its no-data entry. */
export function getDelayTable(floor: number, underFloorCount?: number): ClassTable {
  return makeClassTable({
    breaks: DELAY_BREAKS,
    colors: getDelayColors(),
    unit: 'min',
    extent: [0, 60],
    method: 'Mean departure delay per pair, flights-weighted (a mean, not a median)',
    noData: {
      label: `Under ${floor} flights`,
      color: UNDER_FLOOR_COLOR,
      ...(underFloorCount === undefined ? {} : {count: underFloorCount})
    }
  });
}

/** Class index of a value against ascending interior breaks. */
export function getClassOf(value: number, breaks: readonly number[]): number {
  let index = 0;
  while (index < breaks.length && value >= breaks[index]) index++;
  return index;
}

/**
 * The eight-entry palette of the route layer for the active colour mode, as RGBA 0-255 with the
 * alpha of each slot (`brightness` scales every alpha). Slots 0-4 are classes. Without probes,
 * slots 5-7 are under-floor, single-colour and ghost; with probes, those three otherwise-unused
 * world-network slots become the three probe colours.
 */
export function getRoutePalette(
  colorBy: 'plain' | 'distance' | 'delay',
  brightness: number,
  probesActive: boolean
): PaletteColor[] {
  const withAlpha = (color: PaletteColor, alpha: number): PaletteColor => [
    color[0],
    color[1],
    color[2],
    Math.round(Math.min(1, Math.max(0, alpha)) * 255)
  ];
  const palette: PaletteColor[] = Array.from({length: 8}, () => [0, 0, 0, 0] as PaletteColor);
  const classColors = colorBy === 'delay' ? getDelayColors() : getDistanceColors();
  const classAlpha = colorBy === 'delay' ? DELAY_ALPHA : DISTANCE_ALPHA;
  const dim = probesActive ? PROBE_DIM : 1;
  classColors.forEach((color, index) => {
    palette[SLOT.firstClass + index] = withAlpha(color, classAlpha[index] * brightness * dim);
  });
  if (colorBy === 'plain' && probesActive) {
    palette[SLOT.firstClass] = withAlpha(PLAIN_COLOR, PLAIN_ALPHA * brightness * dim);
  }
  palette[SLOT.underFloor] = withAlpha(UNDER_FLOOR_COLOR, UNDER_FLOOR_ALPHA * brightness);
  palette[SLOT.plain] = withAlpha(PLAIN_COLOR, PLAIN_ALPHA * brightness * dim);
  palette[SLOT.ghost] = withAlpha(UNDER_FLOOR_COLOR, GHOST_ALPHA);
  if (probesActive) {
    PROBE_ROUTES.forEach((probe, index) => {
      palette[SLOT.firstProbe + index] = withAlpha(getProbeColor(probe), 0.95);
    });
  }
  return palette;
}
