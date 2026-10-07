// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * The movement chapter's visual language in one place (design sheet
 * `FID/design/_movement-chapter.md`). Every story of the chapter imports its inks, class tables,
 * credits and time formatters from here, so the same meaning gets the same colour in every story.
 *
 * - **The stage is the same, the subject changes colour.** Tracks, trails and density glow on the
 *   `night` ground; classed fills, proportional symbols and chart-like panels sit on paper.
 * - **One meaning, one scheme.** Vessel groups are four Okabe-Ito hues plus a neutral "other"
 *   ({@link getVesselGroupPalette}); dense vessel views are one uniform amber
 *   ({@link SUBJECT_INK}); ship speed is a classed sequential with a neutral "stopped" class
 *   ({@link getShipSpeedClasses}); waiting is the registry's `waiting` row; machine traffic
 *   density is the registry's `load` row and bird density the `nature` row; bird species are
 *   three Okabe-Ito hues ({@link getSpeciesInk}); aircraft direction and along-track wind share
 *   PuOr (orange = eastbound / tailwind, purple = westbound / headwind, {@link getFlightDirectionInk}).
 * - **Context is one neutral ink** ({@link CONTEXT_TRACK_INK}); selection is achromatic
 *   (`SELECTION_INK`); time windows and probes are `--map-signal`.
 */

import {CREDITS} from '../../cartography/credits';
import {getClassPalette, hexToRgba, rgbToHex} from '../../cartography/class-table';
import {
  getRegistryColors,
  OKABE_ITO_DARK_HEXES,
  OKABE_ITO_LIGHT_HEXES,
  OTHER_GREY,
  type GroundPair,
  type MapGround
} from '../../cartography/hue-registry';
import {sampleRamp, type PaletteColor} from '../../engine/ramps';
import type {LegendSpec} from '../scene';

// ---------------------------------------------------------------------------------------------
// Credits and standing sample lines
// ---------------------------------------------------------------------------------------------

/** Source credits (with licences) of the chapter's datasets; pass them to `joinCredits`. */
export const MOVEMENT_CREDITS = {
  harborAis: 'NOAA OCM / USCG Nationwide AIS, 12 June 2024 (CC0 1.0)',
  harborZones: 'NOAA ENC anchorages and channels (public domain); six areas hand-drawn',
  usAis: 'NOAA / BOEM / USCG Marine Cadastre AIS, 9 January 2023 (public domain)',
  birds: 'GPS tracking: INBO via GBIF (CC0 1.0)',
  gulls: 'GPS tracking: LifeWatch / INBO, UvA-BiTS (Stienen et al., CC0 1.0)',
  openSky: CREDITS.openSky
} as const;

/** The day and place of each dataset, for cartouche subtitles (counts come from the data). */
export const MOVEMENT_SAMPLE_FRAMES = {
  harbor: 'New York Harbor, Wednesday 12 June 2024',
  us: 'US coastal and inland waters, Monday 9 January 2023 (UTC)',
  birds: 'Tagged in Flanders and the Netherlands; years folded onto one calendar',
  gulls: 'North Sea coast colonies, 15 July to 30 November 2015',
  flights: 'Contiguous US, Monday 6 January 2020 (UTC)'
} as const;

/**
 * The unit-of-count caveat every bird story carries in its cartouche: an animal-year is one bird
 * in one tagged year, so the same bird can appear more than once.
 */
export const ANIMAL_YEAR_NOTE =
  'An animal-year is one bird in one tagged year: the same bird can appear more than once.';

// ---------------------------------------------------------------------------------------------
// Inks
// ---------------------------------------------------------------------------------------------

/**
 * The subject of a dense moving view (every vessel or aircraft at once): one warm amber, the
 * shipmap.org default. Hue is opt-in (vessel groups) once the reader asks "what kind?".
 */
export const SUBJECT_INK: GroundPair<PaletteColor> = {
  dark: hexToRgba('#ffd27a'),
  light: hexToRgba('#7a4b00')
};

/**
 * Every track of the dataset as context: one neutral ink, 0.6-0.8 px. On the night ground draw it
 * with additive blending so busy lanes brighten (the alpha is low on purpose); on paper normal.
 */
export const CONTEXT_TRACK_INK: GroundPair<PaletteColor> = {
  dark: [200, 210, 235, 22],
  light: [40, 50, 70, 34]
};

/** Halo or outline under heads and markers: the ground colour. */
export const HEAD_HALO_INK: GroundPair<PaletteColor> = {
  dark: [10, 13, 18, 220],
  light: [255, 255, 255, 235]
};

/** The home-range ring of the waiting colour scheme: stays too long to be a stop (hollow). */
export const HOME_RANGE_RING: PaletteColor = hexToRgba('#9a9890');

/** Zone enter / exit events (a colour-blind-safe sky / orange pair, not green / red). */
export const ZONE_EVENT_INK = {
  enter: {dark: hexToRgba('#7ccbf5'), light: hexToRgba('#0072b2')} as GroundPair<PaletteColor>,
  exit: {dark: hexToRgba('#ffc247'), light: hexToRgba('#d55e00')} as GroundPair<PaletteColor>
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
// Vessel groups (nominal)
// ---------------------------------------------------------------------------------------------

/** The four vessel groups every ship story uses (AIS type codes grouped by what they do). */
export const VESSEL_GROUPS = ['passenger', 'cargo-tanker', 'tug-tow', 'other'] as const;

/** One of {@link VESSEL_GROUPS}. */
export type VesselGroup = (typeof VESSEL_GROUPS)[number];

/** Legend labels of {@link VESSEL_GROUPS}. */
export const VESSEL_GROUP_LABELS: Record<VesselGroup, string> = {
  passenger: 'Passenger and ferry',
  'cargo-tanker': 'Cargo and tanker',
  'tug-tow': 'Tug and tow',
  other: 'Recreational, fishing, other'
};

/**
 * Group index of each `ais-vessels` category, in its manifest order
 * (`cargo tanker passenger tug fishing pleasure other`).
 */
export const HARBOR_CATEGORY_TO_GROUP: readonly number[] = [1, 1, 0, 2, 3, 3, 3];

/**
 * Group index of each `poopdeck-ais-us` vessel type, in its manifest order
 * (`towing passenger other cargo special tanker fishing`).
 */
export const US_TYPE_TO_GROUP: readonly number[] = [2, 0, 3, 1, 3, 1, 3];

/**
 * Colours of {@link VESSEL_GROUPS} in order: Okabe-Ito blue, yellow-orange and bluish green
 * (lifted on dark grounds) and the neutral "other" grey, which recedes.
 */
export function getVesselGroupPalette(ground: MapGround, alpha = 255): PaletteColor[] {
  const hexes = ground === 'dark' ? OKABE_ITO_DARK_HEXES : OKABE_ITO_LIGHT_HEXES;
  return [
    hexToRgba(hexes[0], alpha),
    hexToRgba(hexes[4], alpha),
    hexToRgba(hexes[2], alpha),
    hexToRgba(OTHER_GREY[ground], alpha)
  ];
}

/**
 * Expands the group palette into a per-category palette (the layers index colours by the
 * dataset's own category), using a category-to-group table such as
 * {@link HARBOR_CATEGORY_TO_GROUP}.
 */
export function getCategoryPaletteFromGroups(
  categoryToGroup: readonly number[],
  ground: MapGround,
  alpha = 255
): PaletteColor[] {
  const groups = getVesselGroupPalette(ground, alpha);
  return categoryToGroup.map(group => groups[group]);
}

/** A categories legend of the four vessel groups (optionally with counts per group). */
export function getVesselGroupLegend(
  ground: MapGround,
  counts?: readonly number[],
  note?: string
): LegendSpec {
  const colors = getVesselGroupPalette(ground);
  return {
    kind: 'categories',
    title: 'Vessel group',
    entries: VESSEL_GROUPS.map((group, index) => ({
      color: colors[index],
      label: VESSEL_GROUP_LABELS[group],
      ...(counts ? {count: counts[index]} : {})
    })),
    note: note ?? 'AIS ship-type codes grouped by what the vessel does.'
  };
}

// ---------------------------------------------------------------------------------------------
// Ship speed (classed)
// ---------------------------------------------------------------------------------------------

/** Ship speed class breaks in knots: stopped, manoeuvring, slow, cruising, fast. */
export const SHIP_SPEED_BREAKS_KNOTS: readonly number[] = [0.5, 3, 8, 15];

/** Labels of the five ship speed classes. */
export const SHIP_SPEED_LABELS: readonly string[] = [
  'Stopped (under 0.5 kn)',
  '0.5-3 kn',
  '3-8 kn',
  '8-15 kn',
  'Over 15 kn'
];

/**
 * Five ship speed colours, low class first. The stopped class is the neutral "other" grey (a
 * stopped vessel is drawn as a square, so shape already says it); the moving classes rise in
 * lightness on dark grounds (magma, trimmed so the slowest stays visible) and in darkness on
 * paper (YlGnBu without its near-white end).
 */
export function getShipSpeedClasses(ground: MapGround, alpha = 255): PaletteColor[] {
  const stopped = hexToRgba(OTHER_GREY[ground], alpha);
  const moving =
    ground === 'dark'
      ? [0.42, 0.62, 0.8, 0.97].map(t => sampleRamp('magma', t))
      : getClassPalette('YlGnBu', 6, {ground}).slice(2);
  return [stopped, ...moving.map(color => [color[0], color[1], color[2], alpha] as PaletteColor)];
}

/** A classes legend of the ship speed classes (from the same table as the layer). */
export function getShipSpeedLegend(ground: MapGround, note?: string): LegendSpec {
  const colors = getShipSpeedClasses(ground);
  return {
    kind: 'classes',
    title: 'Speed over ground',
    unit: 'kn',
    breaks: SHIP_SPEED_BREAKS_KNOTS,
    colors,
    labels: SHIP_SPEED_LABELS,
    layout: 'list',
    noData: {label: 'No position (in a data gap)'},
    note: note ?? 'Squares are stopped vessels; arrows point along the heading.'
  };
}

// ---------------------------------------------------------------------------------------------
// Waiting and density (hue registry rows)
// ---------------------------------------------------------------------------------------------

/** Waiting / dwell classes: the registry's `waiting` row (YlOrRd on paper, magma on dark). */
export function getWaitingClasses(ground: MapGround, classCount = 5): PaletteColor[] {
  return getRegistryColors('waiting', ground, classCount);
}

/** Machine traffic density (ships, aircraft): the registry's `load` row. */
export function getTrafficDensityClasses(ground: MapGround, classCount = 5): PaletteColor[] {
  return getRegistryColors('load', ground, classCount);
}

/** Bird track density: the registry's `nature` row (YlGnBu on paper, inferno on dark). */
export function getBirdDensityClasses(ground: MapGround, classCount = 5): PaletteColor[] {
  return getRegistryColors('nature', ground, classCount);
}

// ---------------------------------------------------------------------------------------------
// Bird species (nominal)
// ---------------------------------------------------------------------------------------------

/** Species order of `poopdeck-animals` (manifest order). */
export const SPECIES_KEYS = ['marsh-harrier', 'montagus-harrier', 'spoonbill'] as const;

/** Common names of {@link SPECIES_KEYS}. */
export const SPECIES_NAMES: readonly string[] = [
  'Western marsh harrier',
  "Montagu's harrier",
  'Eurasian spoonbill'
];

const SPECIES_HEXES: GroundPair<readonly string[]> = {
  dark: ['#56b4e9', '#e69f00', '#cc79a7'],
  light: ['#0072b2', '#d55e00', '#a8467f']
};

/** Okabe-Ito species ink (sky blue, orange, reddish purple), lifted on dark grounds. */
export function getSpeciesInk(ground: MapGround, alpha = 255): PaletteColor[] {
  return SPECIES_HEXES[ground].map(hex => hexToRgba(hex, alpha));
}

/** A categories legend of the three species, with optional animal-year counts. */
export function getSpeciesLegend(
  ground: MapGround,
  counts?: readonly number[],
  note?: string
): LegendSpec {
  const colors = getSpeciesInk(ground);
  return {
    kind: 'categories',
    title: 'Species',
    entries: SPECIES_NAMES.map((name, index) => ({
      color: colors[index],
      label: name,
      ...(counts ? {count: counts[index]} : {})
    })),
    note: note ?? ANIMAL_YEAR_NOTE
  };
}

// ---------------------------------------------------------------------------------------------
// Aircraft direction and along-track wind (PuOr)
// ---------------------------------------------------------------------------------------------

/**
 * Aircraft direction ink, shared by the two flight stories: eastbound is the orange arm of PuOr
 * and westbound the purple arm, the same two hues the balance map and the wind map use (orange =
 * more eastbound / tailwind, purple = more westbound / headwind).
 */
export function getFlightDirectionInk(ground: MapGround): {
  eastbound: PaletteColor;
  westbound: PaletteColor;
} {
  const palette = getClassPalette('PuOr', 7, {ground});
  return {westbound: palette[0], eastbound: palette[6]};
}

/**
 * The signed PuOr table of the flight stories, low class first: purple = below the midpoint
 * (westbound / headwind), orange = above (eastbound / tailwind), with a real neutral class.
 */
export function getFlightDivergingClasses(
  ground: MapGround,
  classCount: 5 | 7 = 7,
  alpha = 255
): PaletteColor[] {
  return getClassPalette('PuOr', classCount, {ground, alpha});
}

// ---------------------------------------------------------------------------------------------
// Ordered time classes (one-way dates)
// ---------------------------------------------------------------------------------------------

/**
 * `count` colours of an ordered date (a one-way season, not a cycle): Crameri batlow, perceptually
 * uniform and readable under colour-vision deficiency, trimmed so neither end merges with the
 * ground. For a folded calendar (day of year) use the registry's cyclic `direction` row instead.
 */
export function getOrderedDateClasses(
  ground: MapGround,
  count: number,
  alpha = 255
): PaletteColor[] {
  const range: readonly [number, number] = ground === 'dark' ? [0.25, 0.95] : [0.08, 0.8];
  return Array.from({length: count}, (_, index) => {
    const color = sampleRamp('batlow', count <= 1 ? 1 : index / (count - 1), false, range);
    return [color[0], color[1], color[2], alpha] as PaletteColor;
  });
}

/** CSS hex of a palette colour (for SVG diagrams and chart series). */
export function toHex(color: PaletteColor): string {
  return rgbToHex(color);
}

// ---------------------------------------------------------------------------------------------
// Units and time
// ---------------------------------------------------------------------------------------------

/** Knots per metre per second. */
export const KNOTS_PER_METRE_SECOND = 3600 / 1852;

/** `12.3 kn` (one decimal under 10 kn, whole knots above). */
export function formatKnots(knots: number): string {
  if (!Number.isFinite(knots)) return '–';
  return `${knots < 10 ? knots.toFixed(1) : Math.round(knots).toLocaleString('en-US')} kn`;
}

/** Metres to `1.4 nmi` (nautical miles, one decimal under 10). */
export function formatNauticalMiles(meters: number): string {
  if (!Number.isFinite(meters)) return '–';
  const miles = meters / 1852;
  return `${miles < 10 ? miles.toFixed(1) : Math.round(miles).toLocaleString('en-US')} nmi`;
}

/** Seconds to a dwell: `45 min`, `3 h 10 min`, `2.4 days`. */
export function formatDwell(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '–';
  if (seconds < 3600) return `${Math.max(1, Math.round(seconds / 60))} min`;
  if (seconds < 48 * 3600) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.round((seconds - hours * 3600) / 60);
    return minutes === 0 || minutes === 60
      ? `${hours + (minutes === 60 ? 1 : 0)} h`
      : `${hours} h ${minutes} min`;
  }
  return `${(seconds / 86400).toFixed(1)} days`;
}

/**
 * Wall-clock time of an offset from a UTC origin in a named zone, for example
 * `formatZonedClock(68400, '2024-06-12T00:00:00Z', 'America/New_York')` gives `3:00 pm EDT`.
 */
export function formatZonedClock(
  secondsFromOrigin: number,
  originIso: string,
  timeZone: string
): string {
  const instant = new Date(Date.parse(originIso) + secondsFromOrigin * 1000);
  return new Intl.DateTimeFormat('en-US', {
    hour: 'numeric',
    minute: '2-digit',
    timeZone,
    timeZoneName: 'short'
  })
    .format(instant)
    .replace(' AM', ' am')
    .replace(' PM', ' pm');
}

/** Day of the (2024, leap) folded year to `14 Oct`. */
export function formatFoldedDay(day: number): string {
  const instant = new Date(Date.UTC(2024, 0, 1) + Math.round(day) * 86400000);
  return new Intl.DateTimeFormat('en-GB', {day: 'numeric', month: 'short', timeZone: 'UTC'}).format(
    instant
  );
}

/** First day of each month of the folded (2024) calendar, with a one-letter label. */
export const FOLDED_MONTH_TICKS: readonly {at: number; label: string}[] = Array.from(
  {length: 12},
  (_, month) => ({
    at: Math.round((Date.UTC(2024, month, 1) - Date.UTC(2024, 0, 1)) / 86400000),
    label: 'JFMAMJJASOND'[month]
  })
);
