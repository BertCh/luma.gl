// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Honesty about Web Mercator (G26). The showcase does not ship a projection system: the map is
 * Web Mercator everywhere (`@deck.gl/core` 9.4.0 lacks `CustomProjectionView`), so national and
 * global maps say so. This module computes the distortion, writes the one-line caveat and
 * provides the furniture presets. Pure TypeScript.
 */

import type {FurnitureSpec} from './types';

/** Latitudes are clamped to this magnitude; Mercator scale diverges at the poles. */
const MAX_LATITUDE = 89.9;

function secant(latitude: number): number {
  const clamped = Math.max(-MAX_LATITUDE, Math.min(MAX_LATITUDE, latitude));
  return 1 / Math.cos((clamped * Math.PI) / 180);
}

/**
 * Linear scale factor of Web Mercator at a latitude: `sec(latitude)`. A distance east-west or
 * north-south is drawn this many times longer than on the ground there (1 at the equator).
 *
 * @example
 * getMercatorScaleFactor(60); // 2
 */
export function getMercatorScaleFactor(latitude: number): number {
  return secant(latitude);
}

/**
 * Area scale factor of Web Mercator at a latitude: `sec^2(latitude)`. A region is drawn this
 * many times larger than its true area relative to the equator.
 *
 * @example
 * getMercatorAreaFactor(41.84) / 1; // about 1.80: Cook County is drawn 1.8x too large
 */
export function getMercatorAreaFactor(latitude: number): number {
  return secant(latitude) ** 2;
}

/** The contiguous United States spans about these latitudes (south, north). */
export const CONUS_LATITUDES: readonly [number, number] = [25, 49];

/** Options of {@link mercatorCaveat}. */
export type MercatorCaveatOptions = {
  /** `[lower, upper]` latitudes of the map. Default {@link CONUS_LATITUDES}. */
  latitudes?: readonly [number, number];
  /**
   * What the map shows: `'area'` (a choropleth, sizes), `'distance'` (rings, routes, bars) or
   * `'both'` (default).
   */
  kind?: 'area' | 'distance' | 'both';
};

function formatLatitude(latitude: number): string {
  const rounded = Math.round(Math.abs(latitude) * 10) / 10;
  return `${rounded}° ${latitude < 0 ? 'S' : 'N'}`;
}

function formatFactor(value: number): string {
  return `${(Math.round(value * 10) / 10).toFixed(1)}×`;
}

/**
 * A one-line honesty note for `furniture.caveat`, computed from the latitudes so the number can
 * never drift from the map. The ratio compares the latitude with the larger magnitude to the
 * smaller one.
 *
 * @example
 * mercatorCaveat();
 * // 'Web Mercator: areas at 49° N are drawn 1.9× larger than at 25° N; the scale bar is true only at its latitude.'
 * mercatorCaveat({latitudes: [35, 60], kind: 'distance'});
 * // 'Web Mercator: distances at 60° N are drawn 1.6× longer than at 35° N; the scale bar is true only at its latitude.'
 */
export function mercatorCaveat(options: MercatorCaveatOptions = {}): string {
  const [first, second] = options.latitudes ?? CONUS_LATITUDES;
  const [low, high] = Math.abs(first) <= Math.abs(second) ? [first, second] : [second, first];
  const kind = options.kind ?? 'both';
  const tail = 'the scale bar is true only at its latitude';
  const areaRatio = getMercatorAreaFactor(high) / getMercatorAreaFactor(low);
  const distanceRatio = getMercatorScaleFactor(high) / getMercatorScaleFactor(low);
  const where = `at ${formatLatitude(high)}`;
  const versus = `than at ${formatLatitude(low)}`;
  if (kind === 'area') {
    return `Web Mercator: areas ${where} are drawn ${formatFactor(areaRatio)} larger ${versus}.`;
  }
  if (kind === 'distance') {
    return `Web Mercator: distances ${where} are drawn ${formatFactor(distanceRatio)} longer ${versus}; ${tail}.`;
  }
  return `Web Mercator: areas ${where} are drawn ${formatFactor(areaRatio)} larger ${versus}; ${tail}.`;
}

/**
 * The latitude to pass as `scaleBar.latitude` for a national map: the middle of the bounds
 * (`[west, south, east, north]`), where the bar's error is smallest over the whole map.
 *
 * @example
 * getScaleBarLatitude([-125, 24.5, -66.9, 49.4]); // 36.95
 */
export function getScaleBarLatitude(bounds: readonly [number, number, number, number]): number {
  return (bounds[1] + bounds[3]) / 2;
}

/**
 * Furniture for a national (CONUS) map: a scale bar true at the middle latitude, hidden below
 * zoom 3.5 where it would mislead, and the computed Web Mercator caveat. Spread it into the
 * scene's `furniture` and add the title and credit.
 *
 * @example
 * furniture: {...NATIONAL_FURNITURE, title: true, credit: true}
 */
export const NATIONAL_FURNITURE: FurnitureSpec = {
  scaleBar: {latitude: (CONUS_LATITUDES[0] + CONUS_LATITUDES[1]) / 2, minZoom: 3.5},
  caveat: mercatorCaveat()
};

/**
 * Furniture for a global map: no scale bar (it is meaningless over 100 degrees of latitude) and
 * a caveat that points to geodesic rings (`geodesicCircle` in `reference-geometry.ts`) for any
 * distance claim.
 *
 * @example
 * furniture: {...GLOBAL_FURNITURE, credit: true}
 */
export const GLOBAL_FURNITURE: FurnitureSpec = {
  scaleBar: false,
  caveat: 'Web Mercator inflates areas toward the poles; rings show true ground distance.'
};

/**
 * Converts Web Mercator metres at a latitude to ground metres (`m * cos(latitude)`). Use it when
 * a parameter was set in projected metres and the story talks about ground distance.
 *
 * Mercator metres are what a plain `meters` option measures on a Web Mercator map: at 46 N
 * a "1000 m" radius covers 1000 * cos(46°) = 695 m of ground (the projected metre is 1.44x
 * too long); at 41.9 N it is 745 m (1.34x).
 *
 * @example
 * mercatorMetersToGroundMeters(1000, 41.9); // about 745
 * // Polygon measures: areas computed in Mercator metres are too large by sec^2(latitude).
 * // Cook County at 41.84 N: 1 / mercatorMetersToGroundMeters(1, 41.84) ** 2 is about 1.80,
 * // so divide a Mercator area by getMercatorAreaFactor(41.84) to get the ground area.
 */
export function mercatorMetersToGroundMeters(meters: number, latitude: number): number {
  return meters / secant(latitude);
}

/**
 * Converts ground metres at a latitude to Web Mercator metres (`m / cos(latitude)`), the inverse
 * of {@link mercatorMetersToGroundMeters}.
 *
 * @example
 * groundMetersToMercatorMeters(1000, 46); // about 1440: a true kilometre is 1.44 projected km
 */
export function groundMetersToMercatorMeters(meters: number, latitude: number): number {
  return meters * secant(latitude);
}
