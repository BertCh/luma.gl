// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Zoom arithmetic shared by layers, furniture and scenes. Pure functions, no DOM.
 *
 * deck.gl zoom 0 is a 512-pixel world, so one CSS pixel covers
 * `40,075,016.686 * cos(latitude) / (512 * 2^zoom)` metres (78,271.5 m at the equator at zoom 0).
 * Reports that use 256-pixel tiles (156,543 m) are off by a factor of two.
 */

/** Equatorial circumference of the WGS84 sphere used by Web Mercator, in metres. */
export const EARTH_CIRCUMFERENCE_METERS = 40075016.686;

/** Size of the deck.gl / MapLibre world at zoom 0, in CSS pixels. */
export const WORLD_SIZE_PIXELS = 512;

/**
 * A value that changes with zoom: a constant, or piecewise-linear stops `[[zoom, value], ...]`
 * sorted by zoom (clamped outside the first and last stop), as in MapLibre `interpolate`
 * expressions.
 */
export type ZoomStops = number | readonly (readonly [number, number])[];

/** Ground metres per CSS pixel at `zoom` and `latitude` (degrees) in Web Mercator. */
export function getMetersPerPixel(zoom: number, latitude: number): number {
  return (
    (EARTH_CIRCUMFERENCE_METERS * Math.cos((latitude * Math.PI) / 180)) /
    (WORLD_SIZE_PIXELS * 2 ** zoom)
  );
}

/** CSS pixels covered by `meters` of ground at `zoom` and `latitude`. */
export function metersToPixels(meters: number, zoom: number, latitude: number): number {
  return meters / getMetersPerPixel(zoom, latitude);
}

/** Zoom at which `meters` of ground span `pixels` CSS pixels at `latitude`. */
export function getZoomForMeters(meters: number, pixels: number, latitude: number): number {
  return Math.log2(
    (EARTH_CIRCUMFERENCE_METERS * Math.cos((latitude * Math.PI) / 180) * pixels) /
      (WORLD_SIZE_PIXELS * Math.max(meters, 1e-9))
  );
}

/** Evaluates {@link ZoomStops} at `zoom` (linear between stops, clamped at the ends). */
export function evaluateZoomStops(stops: ZoomStops, zoom: number): number {
  if (typeof stops === 'number') return stops;
  if (stops.length === 0) return 0;
  if (zoom <= stops[0][0]) return stops[0][1];
  const last = stops[stops.length - 1];
  if (zoom >= last[0]) return last[1];
  for (let index = 1; index < stops.length; index++) {
    const [zoomHigh, valueHigh] = stops[index];
    if (zoom <= zoomHigh) {
      const [zoomLow, valueLow] = stops[index - 1];
      const fraction = (zoom - zoomLow) / Math.max(zoomHigh - zoomLow, 1e-9);
      return valueLow + (valueHigh - valueLow) * fraction;
    }
  }
  return last[1];
}

/**
 * Point sizes for dense point clouds by zoom band (SYNTHESIS 1.3): core radius in CSS pixels.
 * z <= 10: 1.2, z 11: 1.8, z 12-13: 2.6, z >= 14: 4.0. Use with `radiusPixels` on
 * `SpatialAnalysisPointLayer` (`radiusZoomStops`) or evaluate it yourself.
 */
export const DENSE_POINT_RADIUS_STOPS: readonly (readonly [number, number])[] = [
  [10, 1.2],
  [11, 1.8],
  [12, 2.6],
  [13.5, 2.6],
  [14, 4]
];
