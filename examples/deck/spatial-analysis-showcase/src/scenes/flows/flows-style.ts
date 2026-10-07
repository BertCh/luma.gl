// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The flows chapter's visual language, in one place (design sheet `FID/design/_flows-chapter.md`).
 *
 * - **Grounds flip with the measurement level.** Luminous quantities (flow volume, density,
 *   tracks) sit on `night`; signed, classed, rate and nominal-region maps sit on `paperCity`;
 *   matrices sit on `paperSheet`.
 * - **One hue per role.** Flow volume is one warm hue ({@link FLOW_INK}); net flow is PuOr with
 *   orange = more leave than arrive and purple = more arrive than leave ({@link getNetFlowPalette});
 *   pickups and drop-offs are the Okabe-Ito orange and sky pair ({@link PICKUP_INK},
 *   {@link DROPOFF_INK}); nominal groups are Okabe-Ito with a neutral grey "other"
 *   ({@link getGroupPalette}); "between groups" is a non-hue ({@link BETWEEN_GROUPS_INK}).
 * - **One width law.** Flows are `0.5 + k sqrt(w / wMax)` px with ONE `wMax` per dataset
 *   ({@link getFlowWidthPixels}), the same law the shared `SpatialAnalysisFlowLayer` draws, so the
 *   width legend ({@link getFlowWidthLegend}) and the map agree.
 *
 * Credits for the chapter's datasets live here too ({@link FLOW_CREDITS}) until the shared
 * `CREDITS` table gains them (requested in the chapter handoff).
 */

import type {PaletteColor} from '../../engine/ramps';
import {getClassPalette, hexToRgba} from '../../cartography/class-table';
import {
  OKABE_ITO_DARK_HEXES,
  OKABE_ITO_LIGHT_HEXES,
  OTHER_GREY,
  type GroundPair,
  type MapGround
} from '../../cartography/hue-registry';
import type {LegendSpec} from '../scene';

// ---------------------------------------------------------------------------------------------
// Credits
// ---------------------------------------------------------------------------------------------

/** Source credits (with licences) for the flows chapter's datasets; pass them to `joinCredits`. */
export const FLOW_CREDITS = {
  chicagoTaxi: 'City of Chicago Taxi Trips 2023 (City of Chicago Data Portal)',
  lodes: 'US Census Bureau LEHD LODES 2021 (public domain)',
  nycTaxi: 'NYC TLC yellow-taxi trip records, January 2015 (NYC Open Data)',
  osrmRoutes: 'Routes: OSRM on © OpenStreetMap contributors (ODbL)',
  openFlights: 'OpenFlights.org (ODbL 1.0), routes frozen June 2014',
  bts: 'US BTS On-Time Performance, July 2023 (public domain)',
  bixi: 'BIXI Montréal open data (CC BY, attribution BIXI Montréal)',
  montrealBoroughs: 'Ville de Montréal, limites administratives (CC BY 4.0)'
} as const;

// ---------------------------------------------------------------------------------------------
// Inks
// ---------------------------------------------------------------------------------------------

/**
 * Flow volume (magnitude). On the night ground a luminous gold that brightens where flows stack
 * (additive); on paper the map ink, drawn normally over a ground halo, so a classed choropleth
 * underneath keeps its hues.
 */
export const FLOW_INK: GroundPair<PaletteColor> = {
  dark: [255, 200, 87, 235],
  light: [31, 41, 51, 215]
};

/** The heaviest flows on the night ground: a hotter gold for the top class or the selected flow. */
export const FLOW_INK_HOT: GroundPair<PaletteColor> = {
  dark: [255, 159, 28, 255],
  light: [17, 24, 39, 255]
};

/** Halo under flows and heads: the ground colour (0.8 px for flows, 1-1.5 px for heads). */
export const FLOW_HALO: GroundPair<PaletteColor> = {
  dark: [10, 13, 18, 200],
  light: [255, 255, 255, 235]
};

/** Pickups / origins (Okabe-Ito orange). Orange is the "leaving" side chapter-wide. */
export const PICKUP_INK: GroundPair<PaletteColor> = {
  dark: hexToRgba('#ffc247'),
  light: hexToRgba('#e69f00')
};

/** Drop-offs / destinations (Okabe-Ito sky blue). */
export const DROPOFF_INK: GroundPair<PaletteColor> = {
  dark: hexToRgba('#7ccbf5'),
  light: hexToRgba('#0072b2')
};

/** Context lines (zone outlines, straight ghosts, all-route backdrops): neutral, thin, faint. */
export const CONTEXT_INK: GroundPair<PaletteColor> = {
  dark: [138, 148, 163, 46],
  light: [74, 86, 99, 40]
};

/** "Between groups" edges: a non-hue (off-white on dark, slate on light), never a group colour. */
export const BETWEEN_GROUPS_INK: GroundPair<PaletteColor> = {
  dark: hexToRgba('#f4f1e8'),
  light: hexToRgba('#4a4f57')
};

/** Picks the ground variant of a {@link GroundPair}. */
export function inkFor<T>(pair: GroundPair<T>, ground: MapGround): T {
  return pair[ground];
}

/** Returns a colour with a new alpha (0-255). */
export function withInkAlpha(color: PaletteColor, alpha: number): PaletteColor {
  return [color[0], color[1], color[2], Math.round(alpha)];
}

// ---------------------------------------------------------------------------------------------
// Net flow (signed): PuOr, orange = more leave, purple = more arrive
// ---------------------------------------------------------------------------------------------

/** What the two arms and the centre of every net-flow legend in the chapter say, in words. */
export const NET_FLOW_WORDS = {
  low: 'More leave than arrive',
  high: 'More arrive than leave',
  midpoint: 'About balanced'
} as const;

/**
 * Classed PuOr for net flow (arrivals minus departures), low class first: the negative arm
 * (more leave) is orange, the positive arm (more arrive) purple, with a real neutral class in the
 * middle (odd counts). The dark ground gets the dark-neutral table with both arms brightening
 * outward. Use the same table for the layer, the legend and the tooltip swatch.
 */
export function getNetFlowPalette(
  classCount: 5 | 7 | 9,
  ground: MapGround,
  alpha = 255
): PaletteColor[] {
  // PuOr is stored purple-first (low = purple); the chapter's sign convention needs orange low.
  return getClassPalette('PuOr', classCount, {ground, reverse: true, alpha});
}

/**
 * Symmetric class breaks around zero for a signed quantity, from the positive steps:
 * `[-s3, -s2, -s1, s1, s2, s3]` for three steps (7 classes, the middle one `|v| < s1`).
 */
export function getSymmetricBreaks(steps: readonly number[]): number[] {
  const positive = [...steps].sort((a, b) => a - b);
  return [...positive.map(step => -step).reverse(), ...positive];
}

// ---------------------------------------------------------------------------------------------
// Nominal groups (continents, communities, boroughs)
// ---------------------------------------------------------------------------------------------

/**
 * The chapter's nominal palette: seven Okabe-Ito hues (light, or lifted for dark grounds) and a
 * neutral grey "other" as the eighth entry. Assign hues by identity (stable across steps, see
 * `cartography/stable-hues`), never by rank.
 */
export function getGroupPalette(ground: MapGround, alpha = 255): PaletteColor[] {
  const hexes = ground === 'dark' ? OKABE_ITO_DARK_HEXES : OKABE_ITO_LIGHT_HEXES;
  return [
    ...hexes.slice(0, 7).map(hex => hexToRgba(hex, alpha)),
    hexToRgba(OTHER_GREY[ground], alpha)
  ];
}

/** Index of the neutral "other" entry in {@link getGroupPalette}. */
export const OTHER_GROUP_INDEX = 7;

// ---------------------------------------------------------------------------------------------
// Flow widths
// ---------------------------------------------------------------------------------------------

/**
 * Width in CSS pixels of a flow of weight `value`, the law `SpatialAnalysisFlowLayer` draws:
 * `0.5 + (maxWidthPixels - 0.5) * sqrt(value / maxValue)` (or linear), clamped to 0.5-12 px.
 */
export function getFlowWidthPixels(
  value: number,
  maxValue: number,
  maxWidthPixels: number,
  scale: 'sqrt' | 'linear' = 'sqrt'
): number {
  const share = Math.min(1, Math.max(0, value / Math.max(maxValue, 1e-20)));
  const t = scale === 'sqrt' ? Math.sqrt(share) : share;
  return Math.min(12, Math.max(0.5, 0.5 + (Math.min(12, maxWidthPixels) - 0.5) * t));
}

/** Rounds a value down to 1, 2 or 5 times a power of ten (legend sample values). */
export function getNiceFlowValue(value: number): number {
  if (!(value > 0)) return 0;
  const power = 10 ** Math.floor(Math.log10(value));
  const mantissa = value / power;
  return (mantissa >= 5 ? 5 : mantissa >= 2 ? 2 : 1) * power;
}

/** Options of {@link getFlowWidthLegend}. */
export type FlowWidthLegendOptions = {
  title: string;
  /** The ONE dataset-wide maximum the layer uses. */
  maxValue: number;
  maxWidthPixels: number;
  color: PaletteColor;
  /** Formats a sample value with its unit, for example `v => `${formatCount(v)} trips``. */
  format: (value: number) => string;
  scale?: 'sqrt' | 'linear';
  note?: string;
};

/**
 * A three-entry line legend for flow widths: nice sample values at about 100 %, 25 % and 4 % of
 * the dataset maximum, drawn with the same law as the layer.
 */
export function getFlowWidthLegend(options: FlowWidthLegendOptions): LegendSpec {
  const {maxValue, maxWidthPixels, color, format, scale = 'sqrt'} = options;
  const samples = [1, 0.25, 0.04]
    .map(share => getNiceFlowValue(maxValue * share))
    .filter((value, index, all) => value > 0 && all.indexOf(value) === index);
  return {
    kind: 'line',
    title: options.title,
    entries: samples.map(value => ({
      color,
      widthPixels: getFlowWidthPixels(value, maxValue, maxWidthPixels, scale),
      label: format(value)
    })),
    note:
      options.note ??
      (scale === 'sqrt'
        ? 'Width grows with the square root of the flow; one scale for every view.'
        : 'Width grows in proportion to the flow; one scale for every view.')
  };
}

// ---------------------------------------------------------------------------------------------
// Flow buffers from a top-K readback
// ---------------------------------------------------------------------------------------------

/** Input of {@link buildFlowArrows}: a `GPUFlowAggregation` top-K list read back to the CPU. */
export type FlowArrowInput = {
  originZones: ArrayLike<number>;
  destinationZones: ArrayLike<number>;
  weights: ArrayLike<number>;
  /** Number of valid rows in the lists. */
  count: number;
  /** Planar metres of each zone centre, `[x, y]` per zone. */
  getZoneCenter: (zone: number) => readonly [number, number] | null;
  /** Draw at most this many (the heaviest). */
  limit: number;
  /** Skip same-zone rows (a flow map cannot draw them; show them as interior circles instead). */
  skipSelf?: boolean;
};

/** Output of {@link buildFlowArrows}: buffers for `SpatialAnalysisFlowLayer` (`flows`, `values`, `ids`). */
export type FlowArrows = {
  /** `x0, y0, x1, y1` planar metres per flow. */
  flows: Float32Array;
  /** One weight per flow (the layer's `values`). */
  weights: Float32Array;
  /** Draw order: ascending weight, heaviest last (the layer's `ids`). */
  order: Uint32Array;
  /** Origin and destination zone per drawn flow (tooltips, labels). */
  originZones: Uint32Array;
  destinationZones: Uint32Array;
  count: number;
};

/**
 * Turns a top-K flow list (heaviest first, as `GPUFlowAggregation` ranks it) into the buffers
 * `SpatialAnalysisFlowLayer` draws. The list is at most a few hundred rows, so this is a small CPU
 * step after the readback the scene already does for its readouts.
 */
export function buildFlowArrows(input: FlowArrowInput): FlowArrows {
  const limit = Math.max(0, Math.min(input.count, input.limit));
  const flows = new Float32Array(limit * 4);
  const weights = new Float32Array(limit);
  const originZones = new Uint32Array(limit);
  const destinationZones = new Uint32Array(limit);
  let count = 0;
  for (let row = 0; row < input.count && count < limit; row++) {
    const origin = input.originZones[row];
    const destination = input.destinationZones[row];
    if (input.skipSelf && origin === destination) continue;
    const start = input.getZoneCenter(origin);
    const end = input.getZoneCenter(destination);
    if (!start || !end) continue;
    flows.set([start[0], start[1], end[0], end[1]], count * 4);
    weights[count] = input.weights[row];
    originZones[count] = origin;
    destinationZones[count] = destination;
    count++;
  }
  const order = Uint32Array.from({length: count}, (_, index) => index).sort(
    (a, b) => weights[a] - weights[b] || a - b
  );
  return {
    flows: flows.subarray(0, count * 4),
    weights: weights.subarray(0, count),
    order,
    originZones: originZones.subarray(0, count),
    destinationZones: destinationZones.subarray(0, count),
    count
  };
}

// ---------------------------------------------------------------------------------------------
// Time labels
// ---------------------------------------------------------------------------------------------

/** `8` -> `08:00`; fractional hours give minutes (`8.5` -> `08:30`). Wraps past 24. */
export function formatClockHour(hour: number): string {
  const wrapped = ((hour % 24) + 24) % 24;
  const whole = Math.floor(wrapped);
  const minutes = Math.round((wrapped - whole) * 60);
  const carry = minutes === 60 ? 1 : 0;
  return `${String((whole + carry) % 24).padStart(2, '0')}:${String(carry ? 0 : minutes).padStart(2, '0')}`;
}

/** `[17, 20]` -> `17:00-20:00`. */
export function formatHourWindow(start: number, end: number): string {
  return `${formatClockHour(start)}-${formatClockHour(end)}`;
}
