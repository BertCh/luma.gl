// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Pure (luma-free) pieces of the sun-and-shadow scene: the option state, the date helpers, the
 * class tables that the paint pass, the legends and the tooltips all read, and the cartouche
 * subtitle. The scene file (gallery) and the compute module share them.
 */

import {getClassTableLegend, hexToRgba, makeClassTable} from '../../cartography/class-table';
import {NO_DATA_COLOR} from '../../cartography/hue-registry';
import type {ClassTable} from '../../cartography/types';
import type {LegendSpec} from '../scene';
import {
  getElevationLegend,
  getShadowColors,
  getShadowLegend,
  makeInsolationTable,
  makeSunHoursTable,
  type TerrainGroundTone
} from './terrain-palettes';

/** What the map draws: the shadow veil, the relief lit by a light, or a whole day integrated. */
export type SunProduct = 'shadow' | 'light' | 'sun-hours' | 'insolation';

/** How the shadow veil is computed: the horizon map, the exact cast, or where the two differ. */
export type SunMethod = 'horizon' | 'cast' | 'difference';

/** Option state of the sun-and-shadow scene. */
export type SunOptions = {
  product: SunProduct;
  method: SunMethod;
  /** `'map'` is the cartographic light (the chapter ground, NW 315); `'sun'` multiplies in the real sun. */
  light: 'map' | 'sun';
  // Time
  dayOfYear: number;
  hour: number;
  /** Hours since 2026-01-01 00:00 UTC of the shown instant; derived from the two above, drives the map clock. */
  instant: number;
  animate: boolean;
  animationSpeed: number;
  twilight: '-0.833' | '-6' | '-12' | '-18';
  refraction: boolean;
  // Sun hours and insolation
  scale: 'june' | 'day';
  display: 'classes' | 'continuous';
  // Horizon map
  /** Draws the search radius around the pinned cell, and the sectors in the sky diagram. */
  showSearch: boolean;
  horizonRadius: '128' | '256' | '512' | '1024' | '1536';
  horizonDirections: '8' | '16' | '32';
  horizonGrowth: number;
  horizonFormat: 'unorm16' | 'float32';
  horizonAlgorithm: 'march' | 'sweep';
  // Cast shadow
  castRadius: '512' | '1024' | '2048' | 'tile';
  // Shadow and light
  softness: number;
  sunIntensity: number;
  ambient: number;
  exposure: number;
  // Irradiance
  directIrradiance: 'meinel' | '1000' | '600';
  diffuseIrradiance: number;
  tableStep: '5' | '10' | '15' | '30';
};

/** Year of the date slider. */
export const YEAR = 2026;

/** Origin of the map clock: the `instant` option counts hours from here. */
export const CLOCK_ORIGIN_ISO = `${YEAR}-01-01T00:00:00Z`;

/** Day of year of 21 December, 21 March and 21 June of {@link YEAR} (a common year). */
export const DATE_PRESETS = {december: 355, march: 80, june: 172} as const;

/** Formats a 1-based day of year as `"Jun 21"`. */
export function formatDay(dayOfYear: number): string {
  const date = new Date(Date.UTC(YEAR, 0, dayOfYear));
  return `${date.toLocaleString('en-US', {month: 'short', timeZone: 'UTC'})} ${date.getUTCDate()}`;
}

/** Formats a 1-based day of year as `"21 Jun"`, the order of the cartouche. */
export function formatDayMonth(dayOfYear: number): string {
  const date = new Date(Date.UTC(YEAR, 0, dayOfYear));
  return `${date.getUTCDate()} ${date.toLocaleString('en-US', {month: 'short', timeZone: 'UTC'})}`;
}

/** Formats decimal hours as `"15:45"`. */
export function formatHour(hours: number): string {
  const totalMinutes = Math.round(hours * 60) % 1440;
  const minutes = totalMinutes % 60;
  return `${String(Math.floor(totalMinutes / 60)).padStart(2, '0')}:${String(minutes).padStart(2, '0')}`;
}

const ZURICH_FORMAT = new Intl.DateTimeFormat('en-US', {
  timeZone: 'Europe/Zurich',
  hourCycle: 'h23',
  year: 'numeric',
  month: 'numeric',
  day: 'numeric',
  hour: 'numeric',
  minute: 'numeric',
  second: 'numeric'
});

/** Offset of Europe/Zurich from UTC in hours (1 in winter, 2 in summer) at a UTC time. */
export function getZurichOffsetHours(utcMilliseconds: number): number {
  const parts = Object.fromEntries(
    ZURICH_FORMAT.formatToParts(new Date(utcMilliseconds)).map(part => [
      part.type,
      Number(part.value)
    ])
  );
  const localAsUtc = Date.UTC(
    parts.year,
    parts.month - 1,
    parts.day,
    parts.hour,
    parts.minute,
    parts.second
  );
  return Math.round((localAsUtc - Math.floor(utcMilliseconds / 1000) * 1000) / 3600000);
}

/** UTC milliseconds of local (Europe/Zurich) civil time on a day of the year. */
export function getUtcFromZurich(dayOfYear: number, hour: number): number {
  const naive = Date.UTC(YEAR, 0, dayOfYear, 0, 0, 0) + hour * 3600000;
  let offset = getZurichOffsetHours(naive - 2 * 3600000);
  offset = getZurichOffsetHours(naive - offset * 3600000);
  return naive - offset * 3600000;
}

/** `"CET"` or `"CEST"` for a UTC time. */
export function getZurichZoneName(utcMilliseconds: number): string {
  return getZurichOffsetHours(utcMilliseconds) === 2 ? 'CEST' : 'CET';
}

/** The value of the derived `instant` option for a local date and time. */
export function getInstantHours(dayOfYear: number, hour: number): number {
  return (getUtcFromZurich(dayOfYear, hour) - Date.UTC(YEAR, 0, 1)) / 3600000;
}

// ---------------------------------------------------------------------------------------------
// Class tables
// ---------------------------------------------------------------------------------------------

/** What the date contributes to the scales: how long the day is and the most direct energy it can deliver. */
export type SunDay = {
  /** Hours the sun is above a flat horizon at Zermatt. */
  dayLengthHours: number;
  /** Direct clear-sky energy in kWh/m² on a surface that faces the sun all day. */
  directEnergyKilowattHours: number;
};

/** A placeholder day (the winter solstice) until the scene has read the real one. */
export const DEFAULT_SUN_DAY: SunDay = {dayLengthHours: 8.4, directEnergyKilowattHours: 4};

/** Clear-sky insolation class breaks in kWh/m² per day on the common (June) scale. */
export const JUNE_INSOLATION_BREAKS: readonly number[] = [1.5, 3, 4.5, 6, 7.5, 9];

/** Longest day, in hours, for which "fit this day" uses the six-class (8 h or more) hours scale. */
const SHORT_DAY_HOURS = 10;

/** Difference of the two shadow methods, as shares of full sun, in OrRd. */
const DIFFERENCE_BREAKS: readonly number[] = [0.02, 0.05, 0.1, 0.2, 0.35];

/**
 * Share of full sun by which the horizon map and the exact cast may differ before a cell counts as
 * "disagreeing" in the `differenceShare` readout (a tenth of the sun's disc).
 */
export const DIFFERENCE_THRESHOLD = 0.1;

/** Breaks that scale the insolation classes to the most energy the day can deliver (0.1 kWh/m² steps). */
function getFittedInsolationBreaks(directEnergy: number): number[] {
  const breaks: number[] = [];
  for (let step = 1; step <= 6; step++) {
    const rounded = Math.round(((directEnergy * step) / 7) * 10) / 10;
    breaks.push(Math.max(rounded, (breaks[breaks.length - 1] ?? 0) + 0.1));
  }
  return breaks.map(value => Number(value.toFixed(1)));
}

/** The class tables of the scene for one ground, date and scale; layer, legend, tooltip and chart read these. */
export type SunTables = {
  ground: TerrainGroundTone;
  hours: ClassTable;
  insolation: ClassTable;
  difference: ClassTable;
};

/**
 * Builds the sun-hours, insolation and difference tables for the state. `scale: 'june'` keeps the
 * same classes on every date (2 h bins to 16 h; 1.5 kWh/m² steps) so two days can be compared;
 * `'day'` fits them to the date: the six-class hours scale on a short day and insolation classes
 * of one seventh of the day's most direct energy.
 */
export function makeSunTables(
  state: Pick<SunOptions, 'scale' | 'dayOfYear'>,
  ground: TerrainGroundTone,
  day: SunDay
): SunTables {
  const fitHours = state.scale === 'day' && day.dayLengthHours <= SHORT_DAY_HOURS;
  const hoursBase = makeSunHoursTable(fitHours ? 'december' : 'june', ground);
  const hours: ClassTable = {
    ...hoursBase,
    method: fitHours
      ? `Fitted to ${formatDayMonth(state.dayOfYear)} (2 h bins to 8 h)`
      : 'Common scale of every date (2 h bins to 16 h)'
  };
  const insolationBreaks =
    state.scale === 'june'
      ? JUNE_INSOLATION_BREAKS
      : getFittedInsolationBreaks(day.directEnergyKilowattHours);
  const insolationBase = makeInsolationTable(insolationBreaks, ground);
  const insolation: ClassTable = {
    ...insolationBase,
    method:
      state.scale === 'june'
        ? 'Common scale of every date (1.5 kWh/m² steps)'
        : `Fitted to ${formatDayMonth(state.dayOfYear)}: one seventh of its direct energy per class`
  };
  const difference = makeClassTable({
    breaks: DIFFERENCE_BREAKS,
    scheme: 'OrRd',
    ground,
    alpha: 200,
    transparent: [0],
    labels: ['Under 2%', '2-5%', '5-10%', '10-20%', '20-35%', '35% or more'],
    unit: 'of full sun',
    method: 'Absolute difference of the two sun visibilities',
    noData: {label: 'No data', color: NO_DATA_COLOR[ground]}
  });
  return {ground, hours, insolation, difference};
}

/** Class tables for the default state on a light ground (legend before the scene reports its own). */
export const DEFAULT_SUN_TABLES: SunTables = makeSunTables(
  {scale: 'june', dayOfYear: DATE_PRESETS.december},
  'light',
  DEFAULT_SUN_DAY
);

/**
 * The anchor value of each class for continuous colour: the middle of its range (the first class
 * is anchored at its lower break, so "no sun" stays at zero). The last class extends one class
 * width above the last break.
 */
export function getTableAnchors(table: ClassTable): number[] {
  const breaks = table.breaks;
  const anchors: number[] = [0];
  for (let index = 1; index <= breaks.length; index++) {
    const low = breaks[index - 1];
    const high = index < breaks.length ? breaks[index] : low + (low - (breaks[index - 2] ?? 0));
    anchors.push((low + high) / 2);
  }
  return anchors;
}

/** Colours of a class table linearly interpolated between the class anchors, `count` evenly spaced stops. */
export function getContinuousColors(
  table: ClassTable,
  anchors: readonly number[],
  count = 24
): [number, number, number, number][] {
  const first = anchors[0];
  const last = anchors[anchors.length - 1];
  return Array.from({length: count}, (_, index) => {
    const value = first + ((last - first) * index) / (count - 1);
    let segment = 0;
    while (segment + 2 < anchors.length && value > anchors[segment + 1]) segment++;
    const low = anchors[segment];
    const high = anchors[segment + 1];
    const position = Math.min(1, Math.max(0, (value - low) / Math.max(high - low, 1e-9)));
    const a = table.colors[segment];
    const b = table.colors[segment + 1];
    const mix = (channel: 0 | 1 | 2 | 3) =>
      Math.round((a[channel] ?? 255) * (1 - position) + (b[channel] ?? 255) * position);
    return [mix(0), mix(1), mix(2), mix(3)];
  });
}

// ---------------------------------------------------------------------------------------------
// Light colours and text
// ---------------------------------------------------------------------------------------------

/** The colours the real sun multiplies into the relief: warm in full sun, cool in the shade. */
export function getLightColors(ground: TerrainGroundTone) {
  return ground === 'dark'
    ? {lit: [1, 1, 1] as const, shade: [0.64, 0.7, 0.86] as const}
    : {lit: [1, 0.985, 0.94] as const, shade: [0.5, 0.57, 0.74] as const};
}

/** Subtitle (line 2 of the cartouche): date, time, zone and what the map shows, live from the options. */
export function getCartoucheSubtitle(
  state: SunOptions,
  cellMeters: number,
  zoneName: string
): string {
  const day = formatDayMonth(state.dayOfYear);
  const cell = `${cellMeters.toFixed(1)} m cells`;
  const time = `${formatHour(state.hour)} ${zoneName}`;
  switch (state.product) {
    case 'shadow':
      return `${day} · ${time} · sun visibility from ${
        state.method === 'horizon'
          ? 'a horizon map'
          : state.method === 'cast'
            ? 'an exact cast'
            : 'both methods, difference'
      } · clear sky`;
    case 'light':
      return state.light === 'map'
        ? `Relief lit from the north-west (map light, 315°) · ${cell}`
        : `${day} · ${time} · relief lit by the real sun · clear sky`;
    case 'sun-hours':
      return `${day} · hours of direct sun · ${state.display === 'classes' ? 'classes' : 'continuous'} · clear sky`;
    case 'insolation':
      return `${day} · clear-sky insolation, kWh/m² per day · ${cell}`;
  }
}

// ---------------------------------------------------------------------------------------------
// Legends
// ---------------------------------------------------------------------------------------------

/** The legends of the current state; tables and counts come from `data` (the ground the scene is on). */
export function getSunLegends(
  state: SunOptions,
  data: Readonly<Record<string, unknown>>
): LegendSpec[] {
  const tables = (data.tables as SunTables | undefined) ?? DEFAULT_SUN_TABLES;
  switch (state.product) {
    case 'shadow': {
      if (state.method === 'difference') {
        return [
          getClassTableLegend(tables.difference, {
            title: 'Where the two methods disagree',
            id: 'difference',
            layout: 'bar',
            note: 'Share of full sun. Cells under 2% are not drawn.'
          })
        ];
      }
      const legend = getShadowLegend(tables.ground);
      return [
        {
          ...legend,
          title: 'Direct sun (clear sky)',
          note: 'Veil opacity follows the share of the sun disc that is hidden.'
        } as LegendSpec
      ];
    }
    case 'light':
      if (state.light === 'map') {
        return [getElevationLegend(tables.ground, {id: 'elevation'})];
      }
      return [
        {
          kind: 'ramp',
          id: 'light',
          title: 'Real sun on the relief',
          ramp: 'grayscale',
          extent: [0, 1],
          labels: ['shaded', 'in sun'],
          colors: [
            [...getLightColors(tables.ground).shade.map(value => Math.round(value * 255))] as [
              number,
              number,
              number
            ],
            [...getLightColors(tables.ground).lit.map(value => Math.round(value * 255))] as [
              number,
              number,
              number
            ]
          ],
          note: 'The relief ground multiplied by sun and sky light. Snow stays below white.'
        }
      ];
    case 'sun-hours':
    case 'insolation': {
      const isHours = state.product === 'sun-hours';
      const table = isHours ? tables.hours : tables.insolation;
      if (state.display === 'continuous') {
        const anchors = getTableAnchors(table);
        return [
          {
            kind: 'ramp',
            id: state.product,
            title: isHours ? 'Hours of direct sun' : 'Clear-sky insolation',
            ramp: 'grayscale',
            colors: getContinuousColors(table, anchors),
            extent: [anchors[0], anchors[anchors.length - 1]],
            unit: isHours ? 'h' : 'kWh/m² per day',
            note: 'Continuous colour hides the class thresholds.'
          }
        ];
      }
      return [
        getClassTableLegend(table, {
          title: isHours ? 'Hours of direct sun' : 'Clear-sky insolation (kWh/m² per day)',
          id: state.product,
          layout: 'bar'
        })
      ];
    }
  }
}

/** The shadow colours of the veil on a ground, re-exported so the compute module needs one import. */
export {getShadowColors};

/** The gold of the sun and the Zermatt eye as an RGBA colour. */
export const SUN_GOLD = hexToRgba('#F2B134');
