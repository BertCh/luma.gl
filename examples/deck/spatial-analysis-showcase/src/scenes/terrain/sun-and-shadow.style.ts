// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Pure (luma-free) pieces of the sun-and-shadow scene: option state, product table and the date
 * helpers shared by the scene file (gallery) and its compute module.
 */

import type {RampName} from '../../engine/ramps';

/** Products of the scene. */
export type SunProduct =
  | 'composite'
  | 'illumination'
  | 'shadow'
  | 'cast-shadow'
  | 'shadow-difference'
  | 'horizon-angle'
  | 'sun-hours'
  | 'insolation'
  | 'sky-view';

/** Option state of the sun-and-shadow scene. */
export type SunOptions = {
  product: SunProduct;
  // Time
  dayOfYear: number;
  hour: number;
  animate: boolean;
  animationSpeed: number;
  twilight: '-0.833' | '-6' | '-12' | '-18';
  refraction: boolean;
  // Shadow
  softness: number;
  ambient: number;
  sunIntensity: number;
  // Horizon map
  horizonDirections: '8' | '16' | '32';
  horizonRadius: '128' | '256' | '384' | '512';
  horizonGrowth: number;
  horizonFormat: 'unorm16' | 'float32';
  horizonAlgorithm: 'march' | 'sweep';
  // Cast shadow
  castRadius: '512' | '1024' | '2048' | 'tile';
  // Irradiance
  directIrradiance: 'meinel' | '1000' | '600';
  diffuseIrradiance: number;
  tableStep: '5' | '10' | '15' | '30';
  // Relief look and display
  reliefShade: number;
  elevationTint: boolean;
  shadowStrength: number;
  lightFloor: number;
  exposure: number;
  ramp: RampName;
  opacity: number;
};

/** Year of the date slider. */
export const YEAR = 2026;

/** Formats a 1-based day of year as `"Jun 21"`. */
export function formatDay(dayOfYear: number): string {
  const date = new Date(Date.UTC(YEAR, 0, dayOfYear));
  return `${date.toLocaleString('en-US', {month: 'short', timeZone: 'UTC'})} ${date.getUTCDate()}`;
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
