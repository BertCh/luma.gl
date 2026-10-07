// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {ViewState} from '../scene';

/** Regions of the earthquake scenes: windows small enough for one planar metric frame. */
export type QuakeRegionId = 'turkiye' | 'japan' | 'andes' | 'sunda' | 'mexico';

/** One analysis window. `bbox` is `[west, south, east, north]` degrees. */
export type QuakeRegion = {
  label: string;
  bbox: readonly [number, number, number, number];
  view: ViewState;
};

/** Windows of up to about 3,000 km, none crossing the antimeridian. */
export const QUAKE_REGIONS: Record<QuakeRegionId, QuakeRegion> = {
  turkiye: {
    label: 'Eastern Mediterranean to Iran',
    bbox: [22, 28, 54, 44],
    view: {longitude: 38, latitude: 36, zoom: 4.7}
  },
  japan: {
    label: 'Japan, Izu-Bonin and the Kurils',
    bbox: [126, 24, 154, 46],
    view: {longitude: 140, latitude: 35, zoom: 4.6}
  },
  andes: {
    label: 'Peru and Chile',
    bbox: [-84, -40, -60, -8],
    view: {longitude: -72, latitude: -24, zoom: 4.4}
  },
  sunda: {
    label: 'Sumatra, Java and Banda',
    bbox: [90, -12, 125, 10],
    view: {longitude: 107.5, latitude: -1, zoom: 4.4}
  },
  mexico: {
    label: 'Mexico and Central America',
    bbox: [-108, 6, -76, 24],
    view: {longitude: -92, latitude: 15, zoom: 4.7}
  }
};

/** Select options for a region control. */
export const QUAKE_REGION_OPTIONS = (Object.keys(QUAKE_REGIONS) as QuakeRegionId[]).map(value => ({
  value,
  label: QUAKE_REGIONS[value].label
}));

/** Unix milliseconds of day 0 of the scenes' clock, 2020-01-01 00:00 UTC. */
export const QUAKE_DAY_ZERO_MS = Date.UTC(2020, 0, 1);
/** Days in 2020 to 2024 (one leap year in 2020 and 2024). */
export const QUAKE_DAY_COUNT = 1827;

/** Days from day 0 of the clock to a UTC date. */
export function getQuakeDay(year: number, month: number, day: number, hour = 0): number {
  return (Date.UTC(year, month - 1, day, hour) - QUAKE_DAY_ZERO_MS) / 86400000;
}

/** `6 Feb 2023` for days since 2020-01-01. */
export function formatQuakeDay(days: number): string {
  const date = new Date(QUAKE_DAY_ZERO_MS + days * 86400000);
  const month = date.toLocaleString('en-GB', {month: 'short', timeZone: 'UTC'});
  return `${date.getUTCDate()} ${month} ${date.getUTCFullYear()}`;
}

/** `6 Feb 2023 10:24 UTC` for days since 2020-01-01. */
export function formatQuakeDateTime(days: number): string {
  const date = new Date(QUAKE_DAY_ZERO_MS + days * 86400000);
  const hours = String(date.getUTCHours()).padStart(2, '0');
  const minutes = String(date.getUTCMinutes()).padStart(2, '0');
  return `${formatQuakeDay(days)} ${hours}:${minutes} UTC`;
}

/** Boundary classes of PB2002 in the dataset's category order. */
export const BOUNDARY_CLASS_NAMES = [
  'Subduction zone',
  'Oceanic convergent',
  'Continental convergent',
  'Oceanic spreading ridge',
  'Continental rift',
  'Oceanic transform',
  'Continental transform'
] as const;

/** Family of each class code: 0 convergent, 1 divergent, 2 transform. */
export const BOUNDARY_FAMILY_OF_CLASS = [0, 0, 0, 1, 1, 2, 2] as const;
/** Names of the three families. */
export const BOUNDARY_FAMILY_NAMES = ['Convergent', 'Divergent', 'Transform'] as const;

/** Depth-class colors, matching chart slots 2, 3 and 4 in each theme. */
export const QUAKE_DEPTH_CLASS_COLORS = {
  dark: [
    [255, 154, 98],
    [78, 209, 181],
    [185, 149, 255]
  ],
  light: [
    [217, 98, 43],
    [23, 143, 122],
    [138, 79, 214]
  ]
} as const;

/** Names of the three depth classes. */
export const QUAKE_DEPTH_CLASS_NAMES = ['Shallow', 'Intermediate', 'Deep'] as const;
