// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {hexToRgba, makeClassTable} from '../../cartography/class-table';
import type {MapGround} from '../../cartography/hue-registry';
import type {ClassTable} from '../../cartography/types';
import type {PaletteColor} from '../../engine/ramps';
import type {LegendSpec} from '../scene';
import {
  CONTEXT_INK,
  DROPOFF_INK,
  FLOW_INK,
  FLOW_INK_HOT,
  getFlowWidthLegend,
  inkFor
} from './flows-style';

/**
 * Symbolisation of the bixi-bundles scene: the gold trunk class table, the qualitative
 * crossing pair, the marks of the rides step and the clock of the routed rides. Pure data and
 * functions with no luma.gl import, so the scene file can use them for its legends.
 *
 * The map is on the night ground in both page themes, so `ctx.ground()` is always `'dark'`; every
 * ink still goes through a ground pair so a light ground would draw too.
 */

/** Narrowest trunk in CSS pixels: the weakest pairs drawn. */
export const TRUNK_WIDTH_MIN_PIXELS = 0.5;
/** Widest trunk in CSS pixels (the busiest pair of all): `0.5 + 5.5 * sqrt(rides / max)`. */
export const TRUNK_WIDTH_MAX_PIXELS = 6;

/**
 * Interior breaks of the rides-per-pair classes. Each class holds twice the rides of the one
 * below, which is the square-root law of the width read in steps: the drawn pairs (the busiest
 * 30,000) all have at least 15 rides, so the lowest class is "under 20".
 */
export const RIDE_CLASS_BREAKS: readonly number[] = [20, 40, 80, 160];

/** Labels of the five ride classes, low first. */
export const RIDE_CLASS_LABELS: readonly string[] = [
  'under 20',
  '20 to 39',
  '40 to 79',
  '80 to 159',
  '160 or more'
];

/** Opacity of each ride class: the same gold, rising with the rides on the pair. */
const RIDE_CLASS_ALPHAS = [0.3, 0.45, 0.62, 0.8, 1] as const;

/**
 * The classed rides-per-pair table: one gold hue (volume is gold chapter-wide) whose alpha rises
 * with the class, so heavy trunks glow and the many weak pairs recede. The layer, the legend and
 * the tooltip read this one table.
 */
export function getRideClassTable(ground: MapGround): ClassTable {
  const base = inkFor(FLOW_INK, ground);
  return makeClassTable({
    breaks: RIDE_CLASS_BREAKS,
    colors: RIDE_CLASS_ALPHAS.map(alpha => [base[0], base[1], base[2], Math.round(alpha * 255)]),
    labels: RIDE_CLASS_LABELS,
    unit: 'rides on the pair',
    method: 'Classes double: each holds twice the rides of the one below. Drawn heaviest last.',
    noData: {label: 'Pair not drawn', color: [138, 148, 163, 60]}
  });
}

/** The qualitative pair of the crossing mode: constant colour per class, not a ramp. */
export const CROSSING_INK: Readonly<
  Record<'within' | 'crossing', {dark: PaletteColor; light: PaletteColor}>
> = {
  within: {dark: hexToRgba('#6e7f99', 184), light: hexToRgba('#5d6b82', 178)},
  crossing: {dark: FLOW_INK_HOT.dark, light: FLOW_INK_HOT.light}
};

/** Palette slots of the crossing mode: 0 stays within a borough, 1 crosses a borough line. */
export function getCrossingPalette(ground: MapGround): PaletteColor[] {
  return [inkFor(CROSSING_INK.within, ground), inkFor(CROSSING_INK.crossing, ground)];
}

/** The straight ghost under the bundles: 0.4 px of the context ink at about 12 % alpha. */
export const GHOST_INK: {dark: PaletteColor; light: PaletteColor} = {
  dark: [CONTEXT_INK.dark[0], CONTEXT_INK.dark[1], CONTEXT_INK.dark[2], 31],
  light: [CONTEXT_INK.light[0], CONTEXT_INK.light[1], CONTEXT_INK.light[2], 28]
};
/** Width of the straight ghost in CSS pixels. */
export const GHOST_WIDTH_PIXELS = 0.4;

/** The bundles in the rides step: the gold at 18 % and 0.5 px, so the bikes are the figure. */
export const DEMOTED_BUNDLE_ALPHA = 46;
/** Width of the demoted bundles in CSS pixels. */
export const DEMOTED_BUNDLE_WIDTH_PIXELS = 0.5;

/** The flat ink of the demoted bundles on a ground. */
export function getDemotedBundleInk(ground: MapGround): PaletteColor {
  const base = inkFor(FLOW_INK, ground);
  return [base[0], base[1], base[2], DEMOTED_BUNDLE_ALPHA];
}

/** Station dots: 1.2 px at 50 % of the soft white of the night ground. */
export const STATION_INK: {dark: PaletteColor; light: PaletteColor} = {
  dark: [207, 214, 228, 128],
  light: [30, 40, 70, 128]
};
/** Radius of a station dot in CSS pixels. */
export const STATION_RADIUS_PIXELS = 1.2;

/** The ring of the six busiest hubs: off-white, so it never competes with the gold. */
export const HUB_RING_INK: {dark: PaletteColor; light: PaletteColor} = {
  dark: [244, 241, 232, 235],
  light: [31, 41, 51, 235]
};
/** Radius of a hub ring in CSS pixels. */
export const HUB_RING_RADIUS_PIXELS = 7;

/** The six busiest stations are ringed; the three busiest are named. */
export const HUB_RING_COUNT = 6;
export const HUB_NAME_COUNT = 3;

/** Trails of the routed rides: the cool sky of the drop-off ink, fading with age. */
export const RIDE_TRAIL_INK: {dark: PaletteColor; light: PaletteColor} = DROPOFF_INK;
/** Width of a ride trail in CSS pixels. */
export const RIDE_TRAIL_WIDTH_PIXELS = 1.6;

/** A bike is a white dot with a dark halo. */
export const BIKE_INK: PaletteColor = [255, 255, 255, 255];
/** Halo of a bike: the night ground. */
export const BIKE_HALO: PaletteColor = [10, 13, 18, 235];
/** Radius of a bike in CSS pixels. */
export const BIKE_RADIUS_PIXELS = 3.5;

/** Trim of the modelled-speed ramp on the night ground. */
export const BIKE_SPEED_RANGE = [0.15, 1] as const;
/** Modelled speed mapped over the ramp, in metres per second. */
export const BIKE_SPEED_EXTENT = [0, 7] as const;

/** A line is "on a street" when a routed ride passes within this many metres of it. */
export const STREET_DISTANCE_METERS = 100;

/**
 * The first instant of the routed rides: 07:30 in Montreal on 15 August 2024, which is 11:30 UTC.
 * `poopdeck-bixi-rides` stores times as seconds since this origin (`properties.timeOriginMs`).
 */
export const RIDES_ORIGIN_MS = Date.UTC(2024, 7, 15, 11, 30);
/** Length of the ride window in seconds: 07:30 to 10:00. */
export const RIDE_WINDOW_SECONDS = 9000;

const CLOCK_FORMAT = new Intl.DateTimeFormat('en-GB', {
  timeZone: 'America/Toronto',
  hour: '2-digit',
  minute: '2-digit',
  hourCycle: 'h23'
});

/** Montreal local time (`HH:MM`, zone `America/Toronto`) of a second of the ride window. */
export function formatRideClock(seconds: number): string {
  return CLOCK_FORMAT.format(new Date(RIDES_ORIGIN_MS + seconds * 1000));
}

/** Legend of the trunk width law: the sample widths are drawn with the same law as the layer. */
export function getTrunkWidthLegend(maxRides: number, ground: MapGround): LegendSpec {
  return getFlowWidthLegend({
    title: 'Trunk width: rides on the pair',
    maxValue: maxRides,
    maxWidthPixels: TRUNK_WIDTH_MAX_PIXELS,
    color: inkFor(FLOW_INK, ground),
    format: value => `${Math.round(value).toLocaleString('en-US')} rides`,
    note: 'Width grows with the square root of the rides; one scale for every step.'
  });
}

/** Legend of the crossing mode: two qualitative classes, with the share of pairs in each. */
export function getCrossingLegend(ground: MapGround, withOutlines = false): LegendSpec {
  const [within, crossing] = getCrossingPalette(ground);
  return {
    kind: 'categories',
    title: 'Where the pair runs',
    entries: [
      {color: within, label: 'Stays inside one borough', shape: 'line'},
      {color: crossing, label: 'Crosses a borough line', shape: 'line'},
      ...(withOutlines
        ? [{color: inkFor(CONTEXT_INK, ground), label: 'Borough outline', shape: 'line' as const}]
        : [])
    ],
    note: 'Two classes, one colour each: a category, not a ramp. Boroughs are as BIXI publishes them.'
  };
}
