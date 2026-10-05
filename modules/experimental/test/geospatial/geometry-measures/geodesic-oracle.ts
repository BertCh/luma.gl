// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * f64 twins of `GEODESIC_WGSL`, used as the oracle by every geometry contributor test. Positions are
 * `[longitude, latitude]` in degrees; angles in radians unless the name says degrees.
 */

const DEGREES_TO_RADIANS = Math.PI / 180;
const RADIANS_TO_DEGREES = 180 / Math.PI;

/** Wraps a longitude difference into `[-180, 180]`. */
export function wrapLongitudeDelta(delta: number): number {
  return delta - 360 * Math.round(delta / 360);
}

/** Unit vector on the sphere for a longitude/latitude pair in degrees. */
export function getUnitVector(lngLat: readonly number[]): [number, number, number] {
  const lambda = lngLat[0] * DEGREES_TO_RADIANS;
  const phi = lngLat[1] * DEGREES_TO_RADIANS;
  return [Math.cos(phi) * Math.cos(lambda), Math.cos(phi) * Math.sin(lambda), Math.sin(phi)];
}

/** Longitude/latitude in degrees of a (not necessarily unit) vector. */
export function getLngLatFromVector(vector: readonly number[]): [number, number] {
  const horizontal = Math.hypot(vector[0], vector[1]);
  return [
    Math.atan2(vector[1], vector[0]) * RADIANS_TO_DEGREES,
    Math.atan2(vector[2], horizontal) * RADIANS_TO_DEGREES
  ];
}

/** Haversine central angle in radians. */
export function getCentralAngle(a: readonly number[], b: readonly number[]): number {
  const deltaPhi = (b[1] - a[1]) * DEGREES_TO_RADIANS;
  const deltaLambda = wrapLongitudeDelta(b[0] - a[0]) * DEGREES_TO_RADIANS;
  const term = Math.min(
    1,
    Math.max(
      0,
      Math.sin(deltaPhi / 2) ** 2 +
        Math.cos(a[1] * DEGREES_TO_RADIANS) *
          Math.cos(b[1] * DEGREES_TO_RADIANS) *
          Math.sin(deltaLambda / 2) ** 2
    )
  );
  return 2 * Math.atan2(Math.sqrt(term), Math.sqrt(1 - term));
}

/** Great-circle distance in `radius` units. */
export function getHaversineDistance(
  a: readonly number[],
  b: readonly number[],
  radius: number
): number {
  return getCentralAngle(a, b) * radius;
}

/** Initial bearing in degrees clockwise from north, in `(-180, 180]` (turf `bearing`). */
export function getInitialBearingDegrees(a: readonly number[], b: readonly number[]): number {
  const phi1 = a[1] * DEGREES_TO_RADIANS;
  const phi2 = b[1] * DEGREES_TO_RADIANS;
  const deltaLambda = wrapLongitudeDelta(b[0] - a[0]) * DEGREES_TO_RADIANS;
  const y = Math.sin(deltaLambda) * Math.cos(phi2);
  const x =
    Math.cos(phi1) * Math.sin(phi2) - Math.sin(phi1) * Math.cos(phi2) * Math.cos(deltaLambda);
  return Math.atan2(y, x) * RADIANS_TO_DEGREES;
}

/** Destination after travelling `angle` radians from `origin` at `bearingDegrees`. */
export function getDestination(
  origin: readonly number[],
  bearingDegrees: number,
  angle: number
): [number, number] {
  const phi1 = origin[1] * DEGREES_TO_RADIANS;
  const theta = bearingDegrees * DEGREES_TO_RADIANS;
  const sinPhi2 =
    Math.sin(phi1) * Math.cos(angle) + Math.cos(phi1) * Math.sin(angle) * Math.cos(theta);
  const phi2 = Math.asin(Math.min(1, Math.max(-1, sinPhi2)));
  const deltaLambda = Math.atan2(
    Math.sin(theta) * Math.sin(angle) * Math.cos(phi1),
    Math.cos(angle) - Math.sin(phi1) * sinPhi2
  );
  return [origin[0] + deltaLambda * RADIANS_TO_DEGREES, phi2 * RADIANS_TO_DEGREES];
}

/**
 * Slerp at `fraction` of the great circle from `a` to `b`, with the longitude unwrapped to stay
 * continuous with `a`.
 */
export function interpolateGreatCircle(
  a: readonly number[],
  b: readonly number[],
  fraction: number
): [number, number] {
  const angle = getCentralAngle(a, b);
  const unitA = getUnitVector(a);
  const unitB = getUnitVector(b);
  let vector: number[];
  if (angle < 1e-12) {
    vector = unitA.map((value, index) => value + (unitB[index] - value) * fraction);
  } else {
    const weightA = Math.sin((1 - fraction) * angle) / Math.sin(angle);
    const weightB = Math.sin(fraction * angle) / Math.sin(angle);
    vector = unitA.map((value, index) => value * weightA + unitB[index] * weightB);
  }
  const lngLat = getLngLatFromVector(vector);
  return [a[0] + wrapLongitudeDelta(lngLat[0] - a[0]), lngLat[1]];
}
