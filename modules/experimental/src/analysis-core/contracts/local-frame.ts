// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_EARTH_MEAN_RADIUS_METERS,
  GPU_WGS84_FLATTENING,
  GPU_WGS84_SEMI_MAJOR_AXIS_METERS
} from './earth';
import {GPU_PLANAR_SPATIAL_CONTEXT, type GPUSpatialContext} from './spatial-context';

/**
 * Projection of a local frame.
 * - `'azimuthal-equidistant'`: spherical azimuthal equidistant on the mean Earth radius. Distances
 *   and bearings from the origin are exact great-circle values; other distances are within about
 *   0.1% up to 200 km from the origin. The frame for distance bands, neighbor radii and buffers.
 * - `'lambert-equal-area'`: ellipsoidal (WGS84) Lambert azimuthal equal-area, as
 *   `+proj=laea +ellps=WGS84`. Areas are exact ellipsoidal areas at any scale. The frame for
 *   densities, areal weights and dot sampling.
 */
export type GPULocalFrameKind = 'azimuthal-equidistant' | 'lambert-equal-area';

/** Options of {@link createGPULocalFrame}. */
export type GPULocalFrameProps = {
  /** Frame origin `[longitude, latitude]` in degrees, kept in binary64 on the CPU. */
  origin: readonly [number, number];
  /** Projection of the frame. Defaults to `'azimuthal-equidistant'`. */
  kind?: GPULocalFrameKind;
  /** Sphere radius in meters for `'azimuthal-equidistant'`. Defaults to the mean Earth radius. */
  sphereRadius?: number;
  /** Frame identity carried by `spatialContext.frame.id`. Defaults to one derived from kind and origin. */
  id?: string;
  /**
   * Horizontal datum of the origin and projected positions, copied to spatialContext.datum.
   * The frame id is unchanged by it.
   */
  datum?: string;
};

/**
 * A binary64 CPU-side local planar frame around an origin, in meters.
 *
 * Projecting longitude/latitude into a local frame before upload is the "project first" step that
 * planar spatial-analysis operators require: the origin is subtracted in binary64, so float32
 * frame coordinates keep about 1e-7 of the frame extent (millimeters at 10 km, centimeters at
 * 100 km) instead of the 0.4 to 1 m quantization of float32 longitude/latitude. Pass
 * {@link GPULocalFrame.spatialContext} to the planar operators; it carries the frame identity so
 * data projected around different origins is rejected as incompatible. Use
 * `GPULocalFrameProjection` to project GPU-resident positions without a readback.
 */
export type GPULocalFrame = {
  readonly kind: GPULocalFrameKind;
  /** Frame origin `[longitude, latitude]` in degrees. */
  readonly origin: readonly [number, number];
  /**
   * Planar native context of frame coordinates, `units: 'meters'`, with the frame identity. Its
   * frame kind lets distance operators reject Lambert frames and area operators reject AEQD frames.
   */
  readonly spatialContext: GPUSpatialContext;
  /** Projects a longitude/latitude in degrees to frame meters. Antipodes give NaN. */
  project(longitude: number, latitude: number): [number, number];
  /** Inverse of {@link project}; longitudes are returned in [-180, 180]. */
  unproject(x: number, y: number): [number, number];
  /**
   * Projects rows of `[longitude, latitude, ...]` with `stride` numbers per row (default 2) to
   * interleaved float32 `[x, y]` frame meters, two per row.
   */
  projectArray(lngLats: ArrayLike<number>, stride?: number, target?: Float32Array): Float32Array;
  /** Unprojects interleaved `[x, y]` frame meters to interleaved binary64 longitude/latitude. */
  unprojectArray(xys: ArrayLike<number>, target?: Float64Array): Float64Array;
};

const DEGREES_TO_RADIANS = Math.PI / 180;
const RADIANS_TO_DEGREES = 180 / Math.PI;
const WGS84_ECCENTRICITY_SQUARED = GPU_WGS84_FLATTENING * (2 - GPU_WGS84_FLATTENING);
const WGS84_ECCENTRICITY = Math.sqrt(WGS84_ECCENTRICITY_SQUARED);
/** Below this distance from a pole (radians) the oblique equal-area formulas switch to polar. */
const POLAR_ORIGIN_EPSILON = 1e-10;

/**
 * Creates a binary64 local frame for projecting longitude/latitude positions before planar
 * analysis. See {@link GPULocalFrame}.
 */
export function createGPULocalFrame(props: GPULocalFrameProps): GPULocalFrame {
  const {
    origin,
    kind = 'azimuthal-equidistant',
    sphereRadius = GPU_EARTH_MEAN_RADIUS_METERS
  } = props;
  const [originLongitude, originLatitude] = origin;
  if (
    !Number.isFinite(originLongitude) ||
    !Number.isFinite(originLatitude) ||
    Math.abs(originLatitude) > 90
  ) {
    throw new Error('createGPULocalFrame origin must be a finite longitude/latitude');
  }
  if (!(sphereRadius > 0) || !Number.isFinite(sphereRadius)) {
    throw new Error('createGPULocalFrame sphereRadius must be positive');
  }
  if (props.datum !== undefined && (typeof props.datum !== 'string' || props.datum.trim() === '')) {
    throw new Error('createGPULocalFrame datum must be a non-empty string');
  }
  const equidistant = kind === 'azimuthal-equidistant';
  const transform = equidistant
    ? createAzimuthalEquidistant(originLongitude, originLatitude, sphereRadius)
    : createLambertAzimuthalEqualArea(originLongitude, originLatitude);
  const frameKind = equidistant ? 'local-azimuthal-equidistant' : 'local-lambert-equal-area';
  const radiusSuffix =
    equidistant && sphereRadius !== GPU_EARTH_MEAN_RADIUS_METERS ? `,R=${sphereRadius}` : '';
  const spatialContext: GPUSpatialContext = Object.freeze({
    ...GPU_PLANAR_SPATIAL_CONTEXT,
    units: 'meters',
    ...(props.datum !== undefined ? {datum: props.datum} : {}),
    frame: Object.freeze({
      id: props.id ?? `${frameKind}(${originLongitude},${originLatitude}${radiusSuffix})`,
      kind: frameKind,
      origin: [originLongitude, originLatitude] as [number, number]
    })
  });

  return Object.freeze({
    kind,
    origin: Object.freeze([originLongitude, originLatitude] as const),
    spatialContext,
    project: (longitude: number, latitude: number) => transform.project(longitude, latitude),
    unproject: (x: number, y: number) => transform.unproject(x, y),
    projectArray(lngLats: ArrayLike<number>, stride = 2, target?: Float32Array): Float32Array {
      if (!Number.isInteger(stride) || stride < 2) {
        throw new Error('createGPULocalFrame projectArray stride must be an integer >= 2');
      }
      const rowCount = Math.floor(lngLats.length / stride);
      const output = target ?? new Float32Array(rowCount * 2);
      for (let row = 0; row < rowCount; row++) {
        const [x, y] = transform.project(lngLats[row * stride], lngLats[row * stride + 1]);
        output[row * 2] = x;
        output[row * 2 + 1] = y;
      }
      return output;
    },
    unprojectArray(xys: ArrayLike<number>, target?: Float64Array): Float64Array {
      const output = target ?? new Float64Array(xys.length);
      for (let index = 0; index + 1 < xys.length; index += 2) {
        const [longitude, latitude] = transform.unproject(xys[index], xys[index + 1]);
        output[index] = longitude;
        output[index + 1] = latitude;
      }
      return output;
    }
  });
}

type LocalFrameTransform = {
  project(longitude: number, latitude: number): [number, number];
  unproject(x: number, y: number): [number, number];
};

/** Longitude difference wrapped to [-pi, pi] radians. */
function getLongitudeDelta(longitude: number, originLongitude: number): number {
  const delta = (longitude - originLongitude) % 360;
  return (delta > 180 ? delta - 360 : delta < -180 ? delta + 360 : delta) * DEGREES_TO_RADIANS;
}

function wrapLongitudeDegrees(longitude: number): number {
  const wrapped = ((((longitude + 180) % 360) + 360) % 360) - 180;
  return wrapped === -180 && longitude > 0 ? 180 : wrapped;
}

/**
 * Azimuth components of a point seen from the origin on a sphere of unit radius, without
 * cancellation for nearby points: `east = cos(phi) sin(dLambda)` and
 * `north = cos(phi0) sin(phi) - sin(phi0) cos(phi) cos(dLambda)`, plus the central angle.
 */
function getSphericalAzimuthComponents(
  sinOrigin: number,
  cosOrigin: number,
  latitude: number,
  originLatitude: number,
  longitudeDelta: number
): {east: number; north: number; centralAngle: number} {
  const cosLatitude = Math.cos(latitude);
  const sinHalfLongitude = Math.sin(0.5 * longitudeDelta);
  const east = cosLatitude * Math.sin(longitudeDelta);
  const north =
    Math.sin(latitude - originLatitude) + 2 * sinOrigin * cosLatitude * sinHalfLongitude ** 2;
  const sinHalfLatitude = Math.sin(0.5 * (latitude - originLatitude));
  const haversine = sinHalfLatitude ** 2 + cosOrigin * cosLatitude * sinHalfLongitude ** 2;
  const centralAngle = 2 * Math.atan2(Math.sqrt(haversine), Math.sqrt(Math.max(0, 1 - haversine)));
  return {east, north, centralAngle};
}

/** Inverse azimuthal step: the point at `centralAngle` along azimuth `(x, y)` from the origin. */
function getSphericalPointFromAzimuth(
  sinOrigin: number,
  cosOrigin: number,
  originLongitude: number,
  x: number,
  y: number,
  rho: number,
  centralAngle: number
): [number, number] {
  if (rho === 0) {
    return [originLongitude, Math.asin(sinOrigin) * RADIANS_TO_DEGREES];
  }
  const sinAngle = Math.sin(centralAngle);
  const cosAngle = Math.cos(centralAngle);
  const latitude = Math.asin(
    Math.max(-1, Math.min(1, cosAngle * sinOrigin + (y * sinAngle * cosOrigin) / rho))
  );
  const longitudeDelta = Math.atan2(
    x * sinAngle,
    rho * cosOrigin * cosAngle - y * sinOrigin * sinAngle
  );
  return [
    wrapLongitudeDegrees(originLongitude + longitudeDelta * RADIANS_TO_DEGREES),
    latitude * RADIANS_TO_DEGREES
  ];
}

function createAzimuthalEquidistant(
  originLongitude: number,
  originLatitudeDegrees: number,
  radius: number
): LocalFrameTransform {
  const originLatitude = originLatitudeDegrees * DEGREES_TO_RADIANS;
  const sinOrigin = Math.sin(originLatitude);
  const cosOrigin = Math.cos(originLatitude);
  return {
    project(longitude, latitude) {
      const {east, north, centralAngle} = getSphericalAzimuthComponents(
        sinOrigin,
        cosOrigin,
        latitude * DEGREES_TO_RADIANS,
        originLatitude,
        getLongitudeDelta(longitude, originLongitude)
      );
      const sinAngle = Math.sin(centralAngle);
      if (centralAngle > Math.PI - 1e-12) {
        // The antipode maps to a circle; it has no single image.
        return [NaN, NaN];
      }
      // k = c / sin(c), scale of the azimuthal equidistant projection; 1 at the origin.
      const scale = centralAngle < 1e-8 ? 1 : centralAngle / sinAngle;
      return [radius * scale * east, radius * scale * north];
    },
    unproject(x, y) {
      const rho = Math.hypot(x, y);
      return getSphericalPointFromAzimuth(
        sinOrigin,
        cosOrigin,
        originLongitude,
        x,
        y,
        rho,
        rho / radius
      );
    }
  };
}

/** Snyder's q(phi) for the WGS84 ellipsoid (Map Projections: A Working Manual, eq. 3-12). */
function getAuthalicQ(sinLatitude: number): number {
  const eccentricitySine = WGS84_ECCENTRICITY * sinLatitude;
  return (
    (1 - WGS84_ECCENTRICITY_SQUARED) *
    (sinLatitude / (1 - eccentricitySine * eccentricitySine) -
      (1 / (2 * WGS84_ECCENTRICITY)) * Math.log((1 - eccentricitySine) / (1 + eccentricitySine)))
  );
}

const POLAR_AUTHALIC_Q = getAuthalicQ(1);
const AUTHALIC_RADIUS = GPU_WGS84_SEMI_MAJOR_AXIS_METERS * Math.sqrt(POLAR_AUTHALIC_Q / 2);

/** Authalic latitude (radians) of a geodetic latitude (radians). */
function getAuthalicLatitude(latitude: number): number {
  return Math.asin(Math.max(-1, Math.min(1, getAuthalicQ(Math.sin(latitude)) / POLAR_AUTHALIC_Q)));
}

/** Geodetic latitude of an authalic latitude by Newton iteration on q (Snyder eq. 3-16). */
function getGeodeticLatitude(authalicLatitude: number): number {
  const targetQ = POLAR_AUTHALIC_Q * Math.sin(authalicLatitude);
  if (Math.abs(Math.abs(authalicLatitude) - Math.PI / 2) < 1e-12) {
    return authalicLatitude;
  }
  let latitude = authalicLatitude;
  for (let iteration = 0; iteration < 12; iteration++) {
    const sinLatitude = Math.sin(latitude);
    const cosLatitude = Math.cos(latitude);
    const denominator = 1 - WGS84_ECCENTRICITY_SQUARED * sinLatitude * sinLatitude;
    // dq/dphi = 2 (1 - e^2) cos(phi) / (1 - e^2 sin^2(phi))^2
    const derivative = (2 * (1 - WGS84_ECCENTRICITY_SQUARED) * cosLatitude) / denominator ** 2;
    const step = (getAuthalicQ(sinLatitude) - targetQ) / derivative;
    latitude -= step;
    if (Math.abs(step) < 1e-15) {
      break;
    }
  }
  return latitude;
}

/**
 * Ellipsoidal Lambert azimuthal equal-area (Snyder eqs. 24-13 to 24-30): geodetic latitudes map
 * to authalic latitudes, then the oblique spherical projection on the authalic sphere is scaled by
 * `D` so the scale is true at the origin in both directions.
 */
function createLambertAzimuthalEqualArea(
  originLongitude: number,
  originLatitudeDegrees: number
): LocalFrameTransform {
  const originLatitude = originLatitudeDegrees * DEGREES_TO_RADIANS;
  const polar = Math.PI / 2 - Math.abs(originLatitude) < POLAR_ORIGIN_EPSILON;
  const originAuthalic = getAuthalicLatitude(originLatitude);
  const sinOrigin = polar ? Math.sign(originLatitude) : Math.sin(originAuthalic);
  const cosOrigin = polar ? 0 : Math.cos(originAuthalic);
  const sinGeodeticOrigin = Math.sin(originLatitude);
  // D = a m(phi0) / (Rq cos(beta0)); 1 for a polar origin.
  const scaleD = polar
    ? 1
    : (GPU_WGS84_SEMI_MAJOR_AXIS_METERS * Math.cos(originLatitude)) /
      Math.sqrt(1 - WGS84_ECCENTRICITY_SQUARED * sinGeodeticOrigin * sinGeodeticOrigin) /
      (AUTHALIC_RADIUS * cosOrigin);
  return {
    project(longitude, latitude) {
      const authalicLatitude = getAuthalicLatitude(latitude * DEGREES_TO_RADIANS);
      const {east, north, centralAngle} = getSphericalAzimuthComponents(
        sinOrigin,
        cosOrigin,
        authalicLatitude,
        polar ? Math.sign(originLatitude) * (Math.PI / 2) : originAuthalic,
        getLongitudeDelta(longitude, originLongitude)
      );
      if (centralAngle > Math.PI - 1e-12) {
        return [NaN, NaN];
      }
      // k' = sqrt(2 / (1 + cos c)) = 1 / cos(c / 2)
      const scale = AUTHALIC_RADIUS / Math.cos(0.5 * centralAngle);
      return [scale * scaleD * east, (scale / scaleD) * north];
    },
    unproject(x, y) {
      const sphereX = x / scaleD;
      const sphereY = y * scaleD;
      const rho = Math.hypot(sphereX, sphereY);
      const centralAngle = 2 * Math.asin(Math.min(1, rho / (2 * AUTHALIC_RADIUS)));
      const [longitude, authalicLatitude] = getSphericalPointFromAzimuth(
        sinOrigin,
        cosOrigin,
        originLongitude,
        sphereX,
        sphereY,
        rho,
        centralAngle
      );
      return [
        longitude,
        getGeodeticLatitude(authalicLatitude * DEGREES_TO_RADIANS) * RADIANS_TO_DEGREES
      ];
    }
  };
}
