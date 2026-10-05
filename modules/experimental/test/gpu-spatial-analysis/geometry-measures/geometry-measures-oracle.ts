// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getCentralAngle, wrapLongitudeDelta} from './geodesic-oracle';

const DEGREES_TO_RADIANS = Math.PI / 180;
const RADIANS_TO_DEGREES = 180 / Math.PI;
const WGS84_A = 6378137;
const WGS84_F = 1 / 298.257223563;
const WGS84_B = WGS84_A * (1 - WGS84_F);
const WGS84_E2 = WGS84_F * (2 - WGS84_F);
const WGS84_E = Math.sqrt(WGS84_E2);

/** Nested geometry: features -> rings (or paths) -> `[x, y]` vertices. */
export type NestedFeatures = number[][][][];

/** Flat geometry layout consumed by `GPUGeometryMeasures`. */
export type FlatGeometry = {
  positions: Float32Array;
  ringOffsets: Uint32Array;
  featureRingOffsets: Uint32Array;
};

/** f64 measures of one feature. */
export type FeatureMeasures = {
  length: number;
  signedArea: number;
  area: number;
  centroid: [number, number];
  bounds: [number, number, number, number];
  vertexCount: number;
};

/** Packs nested features into the flat layout (values rounded to f32). */
export function createFlatGeometry(features: NestedFeatures): FlatGeometry {
  const values: number[] = [];
  const ringOffsets = [0];
  const featureRingOffsets = [0];
  for (const feature of features) {
    for (const ring of feature) {
      for (const vertex of ring) {
        values.push(vertex[0], vertex[1]);
      }
      ringOffsets.push(values.length / 2);
    }
    featureRingOffsets.push(ringOffsets.length - 1);
  }
  return {
    positions: new Float32Array(values),
    ringOffsets: new Uint32Array(ringOffsets),
    featureRingOffsets: new Uint32Array(featureRingOffsets)
  };
}

/** Returns nested features with every coordinate rounded to f32, as the GPU sees them. */
export function roundFeatures(features: NestedFeatures): NestedFeatures {
  return features.map(feature =>
    feature.map(ring => ring.map(vertex => [Math.fround(vertex[0]), Math.fround(vertex[1])]))
  );
}

/** f64 Vincenty inverse on WGS84. Returns the distance in meters and convergence. */
export function getVincentyDistance(
  a: readonly number[],
  b: readonly number[]
): {distance: number; converged: boolean} {
  const lambdaDelta = wrapLongitudeDelta(b[0] - a[0]) * DEGREES_TO_RADIANS;
  const u1 = Math.atan((1 - WGS84_F) * Math.tan(a[1] * DEGREES_TO_RADIANS));
  const u2 = Math.atan((1 - WGS84_F) * Math.tan(b[1] * DEGREES_TO_RADIANS));
  const sinU1 = Math.sin(u1);
  const cosU1 = Math.cos(u1);
  const sinU2 = Math.sin(u2);
  const cosU2 = Math.cos(u2);
  let lambda = lambdaDelta;
  let sinSigma = 0;
  let cosSigma = 1;
  let sigma = 0;
  let cosSquaredAlpha = 1;
  let cos2SigmaM = 0;
  for (let iteration = 0; iteration < 200; iteration++) {
    const sinLambda = Math.sin(lambda);
    const cosLambda = Math.cos(lambda);
    sinSigma = Math.hypot(cosU2 * sinLambda, cosU1 * sinU2 - sinU1 * cosU2 * cosLambda);
    if (sinSigma === 0) {
      return {distance: 0, converged: true};
    }
    cosSigma = sinU1 * sinU2 + cosU1 * cosU2 * cosLambda;
    sigma = Math.atan2(sinSigma, cosSigma);
    const sinAlpha = (cosU1 * cosU2 * sinLambda) / sinSigma;
    cosSquaredAlpha = 1 - sinAlpha * sinAlpha;
    cos2SigmaM = cosSquaredAlpha === 0 ? 0 : cosSigma - (2 * sinU1 * sinU2) / cosSquaredAlpha;
    const c = (WGS84_F / 16) * cosSquaredAlpha * (4 + WGS84_F * (4 - 3 * cosSquaredAlpha));
    const previous = lambda;
    lambda =
      lambdaDelta +
      (1 - c) *
        WGS84_F *
        sinAlpha *
        (sigma + c * sinSigma * (cos2SigmaM + c * cosSigma * (-1 + 2 * cos2SigmaM ** 2)));
    if (Math.abs(lambda - previous) < 1e-13) {
      const uSquared = (cosSquaredAlpha * (WGS84_A ** 2 - WGS84_B ** 2)) / WGS84_B ** 2;
      const bigA =
        1 + (uSquared / 16384) * (4096 + uSquared * (-768 + uSquared * (320 - 175 * uSquared)));
      const bigB = (uSquared / 1024) * (256 + uSquared * (-128 + uSquared * (74 - 47 * uSquared)));
      const deltaSigma =
        bigB *
        sinSigma *
        (cos2SigmaM +
          (bigB / 4) *
            (cosSigma * (-1 + 2 * cos2SigmaM ** 2) -
              (bigB / 6) * cos2SigmaM * (-3 + 4 * sinSigma ** 2) * (-3 + 4 * cos2SigmaM ** 2)));
      return {distance: WGS84_B * bigA * (sigma - deltaSigma), converged: true};
    }
  }
  return {distance: NaN, converged: false};
}

function getAuthalicQ(sinLatitude: number): number {
  const es = WGS84_E * sinLatitude;
  return (
    (1 - WGS84_E2) * (sinLatitude / (1 - es * es) + Math.log((1 + es) / (1 - es)) / (2 * WGS84_E))
  );
}

const POLAR_Q = getAuthalicQ(1);

/** WGS84 authalic radius in meters. */
export const WGS84_AUTHALIC_RADIUS = WGS84_A * Math.sqrt(POLAR_Q / 2);

/** Sine of the authalic latitude for a geodetic latitude in degrees. */
export function getSinAuthalicLatitude(latitudeDegrees: number): number {
  return getAuthalicQ(Math.sin(latitudeDegrees * DEGREES_TO_RADIANS)) / POLAR_Q;
}

/** Exact WGS84 area of a longitude/latitude box in square meters. */
export function getWgs84BoxArea(west: number, south: number, east: number, north: number): number {
  return (
    WGS84_AUTHALIC_RADIUS ** 2 *
    (east - west) *
    DEGREES_TO_RADIANS *
    (getSinAuthalicLatitude(north) - getSinAuthalicLatitude(south))
  );
}

/** Exact spherical area of a longitude/latitude box. */
export function getSphericalBoxArea(
  west: number,
  south: number,
  east: number,
  north: number,
  radius: number
): number {
  return (
    radius ** 2 *
    (east - west) *
    DEGREES_TO_RADIANS *
    (Math.sin(north * DEGREES_TO_RADIANS) - Math.sin(south * DEGREES_TO_RADIANS))
  );
}

/**
 * f64 reference of `GPUGeometryMeasures` for one feature, with the same definitions (local
 * origin, implicit ring closure, hole rules, equal-area cylindrical areas and centroids).
 */
export function measureFeature(
  feature: number[][][],
  options: {
    isPolygon: boolean;
    coordinateSystem: 'planar' | 'spherical' | 'wgs84';
    holeRule?: 'winding' | 'first-ring-exterior';
    radius?: number;
  }
): FeatureMeasures {
  const {isPolygon, coordinateSystem} = options;
  const radius = options.radius ?? 6371008.8;
  const isGeographic = coordinateSystem !== 'planar';
  const rings = feature.filter(ring => ring.length > 0);
  const vertexCount = rings.reduce((sum, ring) => sum + ring.length, 0);
  if (vertexCount === 0) {
    return {
      length: 0,
      signedArea: 0,
      area: 0,
      centroid: [NaN, NaN],
      bounds: [NaN, NaN, NaN, NaN],
      vertexCount: 0
    };
  }
  const origin = rings[0][0];
  const getAreaY = (vertex: number[]): number => {
    if (coordinateSystem === 'planar') {
      return vertex[1] - origin[1];
    }
    if (coordinateSystem === 'spherical') {
      return Math.sin(vertex[1] * DEGREES_TO_RADIANS) - Math.sin(origin[1] * DEGREES_TO_RADIANS);
    }
    return getSinAuthalicLatitude(vertex[1]) - getSinAuthalicLatitude(origin[1]);
  };
  const getEdgeLength = (a: number[], b: number[]): number => {
    if (coordinateSystem === 'planar') {
      return Math.hypot(b[0] - a[0], b[1] - a[1]);
    }
    if (coordinateSystem === 'spherical') {
      return getCentralAngle(a, b) * radius;
    }
    return getVincentyDistance(a, b).distance;
  };
  let length = 0;
  let crossSum = 0;
  let momentX = 0;
  let momentY = 0;
  let lineMomentX = 0;
  let lineMomentY = 0;
  let vertexSumX = 0;
  let vertexSumY = 0;
  const bounds = [Infinity, Infinity, -Infinity, -Infinity];
  rings.forEach((ring, ringIndex) => {
    const locals = ring.map(vertex => [vertex[0] - origin[0], vertex[1] - origin[1]]);
    if (isGeographic) {
      locals[0][0] = wrapLongitudeDelta(ring[0][0] - origin[0]);
      for (let vertex = 1; vertex < ring.length; vertex++) {
        locals[vertex][0] =
          locals[vertex - 1][0] + wrapLongitudeDelta(ring[vertex][0] - ring[vertex - 1][0]);
      }
    }
    for (const local of locals) {
      bounds[0] = Math.min(bounds[0], local[0]);
      bounds[1] = Math.min(bounds[1], local[1]);
      bounds[2] = Math.max(bounds[2], local[0]);
      bounds[3] = Math.max(bounds[3], local[1]);
      vertexSumX += local[0];
      vertexSumY += local[1];
    }
    const edgeCount = isPolygon ? ring.length : ring.length - 1;
    let ringCross = 0;
    let ringMomentX = 0;
    let ringMomentY = 0;
    for (let edge = 0; edge < edgeCount; edge++) {
      const next = (edge + 1) % ring.length;
      const edgeLength = getEdgeLength(ring[edge], ring[next]);
      length += edgeLength;
      lineMomentX += edgeLength * 0.5 * (locals[edge][0] + locals[next][0]);
      lineMomentY += edgeLength * 0.5 * (locals[edge][1] + locals[next][1]);
      const scaleX = isGeographic ? DEGREES_TO_RADIANS : 1;
      const x0 = locals[edge][0] * scaleX;
      const x1 = locals[next][0] * scaleX;
      const y0 = getAreaY(ring[edge]);
      const y1 = getAreaY(ring[next]);
      const cross = x0 * y1 - x1 * y0;
      ringCross += cross;
      ringMomentX += (x0 + x1) * cross;
      ringMomentY += (y0 + y1) * cross;
    }
    let factor = 1;
    if (options.holeRule === 'first-ring-exterior') {
      factor = (ringIndex === 0 ? 1 : -1) * (ringCross >= 0 ? 1 : -1);
    }
    crossSum += factor * ringCross;
    momentX += factor * ringMomentX;
    momentY += factor * ringMomentY;
  });
  const areaRadius = coordinateSystem === 'wgs84' ? WGS84_AUTHALIC_RADIUS : radius;
  const signedArea = isPolygon ? 0.5 * crossSum * (isGeographic ? areaRadius ** 2 : 1) : 0;
  let centroid: [number, number] = [
    origin[0] + vertexSumX / vertexCount,
    origin[1] + vertexSumY / vertexCount
  ];
  if (isPolygon && crossSum !== 0) {
    const localX = momentX / (3 * crossSum);
    const localY = momentY / (3 * crossSum);
    if (coordinateSystem === 'planar') {
      centroid = [origin[0] + localX, origin[1] + localY];
    } else if (coordinateSystem === 'spherical') {
      centroid = [
        origin[0] + localX * RADIANS_TO_DEGREES,
        Math.asin(Math.sin(origin[1] * DEGREES_TO_RADIANS) + localY) * RADIANS_TO_DEGREES
      ];
    } else {
      const xi = Math.asin(getSinAuthalicLatitude(origin[1]) + localY);
      const e4 = WGS84_E2 * WGS84_E2;
      const latitude =
        xi +
        (WGS84_E2 / 3 + (31 * e4) / 180) * Math.sin(2 * xi) +
        ((17 * e4) / 360) * Math.sin(4 * xi);
      centroid = [origin[0] + localX * RADIANS_TO_DEGREES, latitude * RADIANS_TO_DEGREES];
    }
  } else if (!isPolygon && length > 0) {
    centroid = [origin[0] + lineMomentX / length, origin[1] + lineMomentY / length];
  }
  return {
    length,
    signedArea,
    area: Math.abs(signedArea),
    centroid,
    bounds: [
      origin[0] + bounds[0],
      origin[1] + bounds[1],
      origin[0] + bounds[2],
      origin[1] + bounds[3]
    ],
    vertexCount
  };
}
