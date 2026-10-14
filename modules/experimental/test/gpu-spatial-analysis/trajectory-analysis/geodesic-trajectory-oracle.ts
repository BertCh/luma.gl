// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Great-circle oracle for the movement operators.
 *
 * `GEODESIC_ORACLE_CASES` are pinned from pyproj 3.8.0 on the sphere `Geod(a=6371008.8, f=0)`
 * (`GPU_EARTH_MEAN_RADIUS_METERS`) by
 * `<scratchpad>/fix/work/E/E3-movement/geodesic_oracle.py` (run with the pysal venv). The CPU
 * functions below are binary64 twins of the WGSL helpers; the node spec checks them against the
 * pinned values and the GPU specs compare the kernels with them.
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

export const GEODESIC_ORACLE_CASES: GeodesicOracleCase[] = [
  {
    name: 'equator-east',
    a: [0, 0],
    b: [10, 0],
    distance: 1111950.8023353291,
    bearingDegrees: 90,
    points: [
      {fraction: 0.25, lon: 2.5, lat: 0},
      {fraction: 0.5, lon: 5, lat: 0},
      {fraction: 0.75, lon: 7.5, lat: 0}
    ]
  },
  {
    name: 'sf-la',
    a: [-122.4194, 37.7749],
    b: [-118.2437, 34.0522],
    distance: 559121.3493517861,
    bearingDegrees: 136.5029105734901,
    points: [
      {fraction: 0.25, lon: -121.33800831725446, lat: 36.85803992814649},
      {fraction: 0.5, lon: -120.28238771442261, lat: 35.93161735278957},
      {fraction: 0.75, lon: -119.25133883300691, lat: 34.996168360115526}
    ]
  },
  {
    name: 'antimeridian',
    a: [179.5, 10],
    b: [-179.5, 12],
    distance: 247730.114027376,
    bearingDegrees: 26.048736460939004,
    points: [
      {fraction: 0.25, lon: 179.74874858764124, lat: 10.500301389960903},
      {fraction: 0.5, lon: 179.99830349647652, lat: 11.000408640099051},
      {fraction: 0.75, lon: 180.24870653787548, lat: 11.500311602028805}
    ]
  },
  {
    name: 'short-step',
    a: [8.5417, 47.3769],
    b: [8.5427, 47.3774],
    distance: 93.59947846884606,
    bearingDegrees: 53.55867351398973,
    points: [
      {fraction: 0.25, lon: 8.541949998222018, lat: 47.3770250008153},
      {fraction: 0.5, lon: 8.542199997629362, lat: 47.37715000108708},
      {fraction: 0.75, lon: 8.542449998222006, lat: 47.3772750008153}
    ]
  }
];

/**
 * Largest searchable distance of the encounter lattice from pyproj: for each case the claimed
 * angle `asin(sin(cell) * cos(L))` was walked in 7201 azimuths from latitude +-L, and the observed
 * longitude and latitude reach stayed within one cell (`observedLongitudeReachDegrees <=
 * cellDegrees`), nearly tight. `maximumDistance` includes the `1 - 1e-6` scale.
 */
export const ENCOUNTER_REACH_ORACLE_CASES = [
  {
    cellDegrees: 2,
    maximumLatitude: 60,
    maximumDistance: 111178.03178664722,
    observedLongitudeReachDegrees: 1.999999903763296
  },
  {
    cellDegrees: 5,
    maximumLatitude: 0,
    maximumDistance: 555974.8451922634,
    observedLongitudeReachDegrees: 5
  },
  {
    cellDegrees: 1,
    maximumLatitude: 80,
    maximumDistance: 19307.853009798393,
    observedLongitudeReachDegrees: 0.9999999648570395
  },
  {
    cellDegrees: 10,
    maximumLatitude: 45,
    maximumDistance: 784260.5774043602,
    observedLongitudeReachDegrees: 9.999999924115855
  }
];

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
