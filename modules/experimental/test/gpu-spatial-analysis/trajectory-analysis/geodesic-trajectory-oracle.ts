// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import fixtureJson from '../../oracles/fixtures/geodesic-trajectory.json';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';

/**
 * Great-circle oracle for the movement operators.
 *
 * `GEODESIC_ORACLE_CASES` are the pyproj references on the sphere `Geod(a=6371008.8, f=0)`
 * (`GPU_EARTH_MEAN_RADIUS_METERS`) written by `test/oracles/geodesic-trajectory.py`. The CPU
 * functions below are binary64 twins of the WGSL helpers; the node spec checks them against the
 * reference values and the GPU specs compare the kernels with them.
 */

export const GEODESIC_ORACLE_RADIUS = 6371008.8;

export type GeodesicOracleCase = {
  name: string;
  a: [number, number];
  b: [number, number];
  /** pyproj `Geod.inv` distance in meters. */
  distance: number;
  /** pyproj `Geod.inv` forward azimuth in degrees, clockwise from north. */
  bearingDegrees: number;
  /** pyproj `Geod.fwd` points at fractions of the distance, longitude continuous with `a`. */
  points: {fraction: number; lon: number; lat: number}[];
};

/** One lattice cell size and latitude bound with the pyproj-walked reach. */
export type EncounterReachOracleCase = {
  cellDegrees: number;
  maximumLatitude: number;
  maximumDistance: number;
  observedLongitudeReachDegrees: number;
};

const fixture = decodeOracleFixture<{
  generator: string;
  versions: Record<string, string>;
  tolerances: Record<'distanceRelative' | 'bearingDegrees' | 'pointDegrees', number>;
  reference: {
    radius: number;
    cases: GeodesicOracleCase[];
    encounterReach: EncounterReachOracleCase[];
  };
}>(fixtureJson);

export const GEODESIC_ORACLE_CASES: GeodesicOracleCase[] = fixture.reference.cases;

/** Tolerances recorded by the generator, with their derivation in the generator docstring. */
export const GEODESIC_ORACLE_TOLERANCES = fixture.tolerances;

/**
 * Largest searchable distance of the encounter lattice from pyproj: for each case the claimed
 * angle `asin(sin(cell) * cos(L))` was walked in 7201 azimuths from latitude +-L, and the observed
 * longitude and latitude reach stayed within one cell (`observedLongitudeReachDegrees <=
 * cellDegrees`), nearly tight. `maximumDistance` includes the `1 - 1e-6` scale.
 */
export const ENCOUNTER_REACH_ORACLE_CASES: EncounterReachOracleCase[] =
  fixture.reference.encounterReach;

const DEGREES = Math.PI / 180;

function wrapLongitudeDelta(delta: number): number {
  return delta - 360 * Math.round(delta / 360);
}

/** Central angle in radians (Vincenty's sphere formula). */
export function getCentralAngle(a: readonly number[], b: readonly number[]): number {
  const phi1 = a[1] * DEGREES;
  const phi2 = b[1] * DEGREES;
  const deltaLambda = wrapLongitudeDelta(b[0] - a[0]) * DEGREES;
  const east = Math.cos(phi2) * Math.sin(deltaLambda);
  const north =
    Math.cos(phi1) * Math.sin(phi2) - Math.sin(phi1) * Math.cos(phi2) * Math.cos(deltaLambda);
  const up =
    Math.sin(phi1) * Math.sin(phi2) + Math.cos(phi1) * Math.cos(phi2) * Math.cos(deltaLambda);
  return Math.atan2(Math.hypot(east, north), up);
}

/** Great-circle distance in the units of `radius`. */
export function getGreatCircleDistance(
  a: readonly number[],
  b: readonly number[],
  radius: number = GEODESIC_ORACLE_RADIUS
): number {
  return radius * getCentralAngle(a, b);
}

/** Initial bearing in degrees clockwise from north, in `(-180, 180]`. */
export function getInitialBearingDegrees(a: readonly number[], b: readonly number[]): number {
  const phi1 = a[1] * DEGREES;
  const phi2 = b[1] * DEGREES;
  const deltaLambda = wrapLongitudeDelta(b[0] - a[0]) * DEGREES;
  const y = Math.sin(deltaLambda) * Math.cos(phi2);
  const x =
    Math.cos(phi1) * Math.sin(phi2) - Math.sin(phi1) * Math.cos(phi2) * Math.cos(deltaLambda);
  return Math.atan2(y, x) / DEGREES;
}

/** Spherical interpolation with the longitude continuous with `a` (unwrapped). */
export function interpolateGreatCircle(
  a: readonly number[],
  b: readonly number[],
  fraction: number
): [number, number] {
  const unit = (point: readonly number[]) => {
    const lambda = point[0] * DEGREES;
    const phi = point[1] * DEGREES;
    return [Math.cos(phi) * Math.cos(lambda), Math.cos(phi) * Math.sin(lambda), Math.sin(phi)];
  };
  const angle = getCentralAngle(a, b);
  const unitA = unit(a);
  const unitB = unit(b);
  const weightA = Math.sin((1 - fraction) * angle) / Math.sin(angle);
  const weightB = Math.sin(fraction * angle) / Math.sin(angle);
  const vector = unitA.map((value, index) => value * weightA + unitB[index] * weightB);
  const longitude = Math.atan2(vector[1], vector[0]) / DEGREES;
  const latitude = Math.atan2(vector[2], Math.hypot(vector[0], vector[1])) / DEGREES;
  return [a[0] + wrapLongitudeDelta(longitude - a[0]), latitude];
}

/** Conservative largest searchable encounter distance of the lattice, see `GPUTrajectoryEncounters`. */
export function getEncounterMaximumDistance(
  cellDegrees: number,
  maximumLatitude: number,
  radius: number = GEODESIC_ORACLE_RADIUS
): number {
  return (
    radius *
    Math.asin(Math.sin(cellDegrees * DEGREES) * Math.cos(maximumLatitude * DEGREES)) *
    (1 - 1e-6)
  );
}
