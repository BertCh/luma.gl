// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Mean Earth radius in meters (IUGG R1), the default sphere radius of the geodesic contributors and of
 * turf's `distance`, `length`, `destination` and `along`.
 */
export const GPU_GEODESIC_MEAN_EARTH_RADIUS = 6371008.8;

/** WGS84 semi-major axis in meters. turf's `area` uses this value as its sphere radius. */
export const GPU_GEODESIC_WGS84_SEMI_MAJOR_AXIS = 6378137;

/** WGS84 flattening. */
export const GPU_GEODESIC_WGS84_FLATTENING = 1 / 298.257223563;

/**
 * Shared f32 spherical geodesy helpers for the geometry contributors (`line-segmentize`,
 * `geometry-measures`, `linear-referencing`). Positions are `vec2<f32>(longitude, latitude)` in
 * degrees; angles returned in radians unless the name says degrees.
 *
 * Precision rules, all measured against the f64 twins in the geometry-measures test oracle:
 * - Differences are taken in degrees first (`b - a`), so nearby points keep their relative
 *   precision instead of subtracting two rounded trigonometric values.
 * - Central angles use the haversine term with an `atan2` finish, which is well conditioned for
 *   both tiny and large separations (not near-antipodal).
 * - Cosines of latitude come from the exact colatitude, and bearings and destinations use
 *   half-angle forms of their cosine-difference terms, so meter-scale moves and polar points do
 *   not cancel catastrophically.
 * - Interpolation slerps unit vectors and falls back to a normalized lerp below `1e-6` rad, where
 *   the slerp weights lose precision.
 * - `geodesicSinLatitudeDelta(lat, lat0)` returns `sin(lat) - sin(lat0)` from the half-angle
 *   product so area sums can subtract a local origin without cancellation.
 *
 * @internal
 */
export const GEODESIC_WGSL = /* wgsl */ `
const GEODESIC_PI: f32 = 3.14159265358979;
const GEODESIC_DEGREES_TO_RADIANS: f32 = 0.0174532925199433;
const GEODESIC_RADIANS_TO_DEGREES: f32 = 57.2957795130823;

fn geodesicWrapLongitudeDelta(delta: f32) -> f32 {
  return delta - 360.0 * round(delta / 360.0);
}

// cos(latitude) as sin(colatitude): 90 - |lat| is exact in f32, so the cosine keeps its relative
// precision near the poles where cos(lat * pi / 180) would not.
fn geodesicCosLatitude(latitudeDegrees: f32) -> f32 {
  return sin((90.0 - abs(latitudeDegrees)) * GEODESIC_DEGREES_TO_RADIANS);
}

fn geodesicUnitVector(lngLat: vec2<f32>) -> vec3<f32> {
  let lambda = lngLat.x * GEODESIC_DEGREES_TO_RADIANS;
  let phi = lngLat.y * GEODESIC_DEGREES_TO_RADIANS;
  let cosPhi = geodesicCosLatitude(lngLat.y);
  return vec3<f32>(cosPhi * cos(lambda), cosPhi * sin(lambda), sin(phi));
}

fn geodesicFromUnitVector(vector: vec3<f32>) -> vec2<f32> {
  let horizontal = sqrt(vector.x * vector.x + vector.y * vector.y);
  return vec2<f32>(
    atan2(vector.y, vector.x) * GEODESIC_RADIANS_TO_DEGREES,
    atan2(vector.z, horizontal) * GEODESIC_RADIANS_TO_DEGREES
  );
}

fn geodesicHaversineTerm(a: vec2<f32>, b: vec2<f32>) -> f32 {
  let deltaPhi = (b.y - a.y) * GEODESIC_DEGREES_TO_RADIANS;
  let deltaLambda = geodesicWrapLongitudeDelta(b.x - a.x) * GEODESIC_DEGREES_TO_RADIANS;
  let sinHalfPhi = sin(0.5 * deltaPhi);
  let sinHalfLambda = sin(0.5 * deltaLambda);
  let term = sinHalfPhi * sinHalfPhi +
    geodesicCosLatitude(a.y) * geodesicCosLatitude(b.y) *
    sinHalfLambda * sinHalfLambda;
  return clamp(term, 0.0, 1.0);
}

fn geodesicCentralAngle(a: vec2<f32>, b: vec2<f32>) -> f32 {
  // Vincenty's special case for the sphere: well conditioned from a millimeter to antipodal,
  // unlike the haversine form whose sqrt(1 - h) loses precision near antipodes.
  let phi1 = a.y * GEODESIC_DEGREES_TO_RADIANS;
  let phi2 = b.y * GEODESIC_DEGREES_TO_RADIANS;
  let deltaPhi = (b.y - a.y) * GEODESIC_DEGREES_TO_RADIANS;
  let deltaLambda = geodesicWrapLongitudeDelta(b.x - a.x) * GEODESIC_DEGREES_TO_RADIANS;
  let sinPhi1 = sin(phi1);
  let cosPhi1 = geodesicCosLatitude(a.y);
  let cosPhi2 = geodesicCosLatitude(b.y);
  let sinHalfLambda = sin(0.5 * deltaLambda);
  let halfVersine = 2.0 * sinHalfLambda * sinHalfLambda;
  let east = cosPhi2 * sin(deltaLambda);
  let north = sin(deltaPhi) + sinPhi1 * cosPhi2 * halfVersine;
  let up = sinPhi1 * sin(phi2) + cosPhi1 * cosPhi2 * (1.0 - halfVersine);
  return atan2(sqrt(east * east + north * north), up);
}

fn geodesicInitialBearingDegrees(a: vec2<f32>, b: vec2<f32>) -> f32 {
  let phi1 = a.y * GEODESIC_DEGREES_TO_RADIANS;
  let phi2 = b.y * GEODESIC_DEGREES_TO_RADIANS;
  let deltaPhi = (b.y - a.y) * GEODESIC_DEGREES_TO_RADIANS;
  let deltaLambda = geodesicWrapLongitudeDelta(b.x - a.x) * GEODESIC_DEGREES_TO_RADIANS;
  let sinHalfLambda = sin(0.5 * deltaLambda);
  let cosPhi2 = geodesicCosLatitude(b.y);
  let y = sin(deltaLambda) * cosPhi2;
  // cos(phi1) sin(phi2) - sin(phi1) cos(phi2) cos(deltaLambda), without cancellation.
  let x = sin(deltaPhi) + 2.0 * sin(phi1) * cosPhi2 * sinHalfLambda * sinHalfLambda;
  return atan2(y, x) * GEODESIC_RADIANS_TO_DEGREES;
}

fn geodesicDestination(origin: vec2<f32>, bearingDegrees: f32, angle: f32) -> vec2<f32> {
  let phi1 = origin.y * GEODESIC_DEGREES_TO_RADIANS;
  let theta = bearingDegrees * GEODESIC_DEGREES_TO_RADIANS;
  let sinPhi1 = sin(phi1);
  let cosPhi1 = geodesicCosLatitude(origin.y);
  let sinHalfAngle = sin(0.5 * angle);
  // sin(phi2) - sin(phi1), from the half-angle form so short moves keep their precision.
  let sinDelta = -2.0 * sinPhi1 * sinHalfAngle * sinHalfAngle + cosPhi1 * sin(angle) * cos(theta);
  let sinPhi2 = clamp(sinPhi1 + sinDelta, -1.0, 1.0);
  let roughPhi2 = asin(sinPhi2);
  // sin(phi2) - sin(phi1) = 2 cos((phi1 + phi2) / 2) sin((phi2 - phi1) / 2); the midpoint cosine
  // is insensitive to the rough phi2, so the latitude change is recovered to relative precision.
  let midpointCosine = cos(0.5 * (phi1 + roughPhi2));
  var deltaPhi = roughPhi2 - phi1;
  if (abs(midpointCosine) > 1e-3) {
    deltaPhi = 2.0 * asin(clamp(sinDelta / (2.0 * midpointCosine), -1.0, 1.0));
  }
  let deltaLambda = atan2(
    sin(theta) * sin(angle) * cosPhi1,
    cos(angle) - sinPhi1 * sinPhi2
  );
  return vec2<f32>(
    origin.x + deltaLambda * GEODESIC_RADIANS_TO_DEGREES,
    origin.y + deltaPhi * GEODESIC_RADIANS_TO_DEGREES
  );
}

fn geodesicInterpolate(a: vec2<f32>, b: vec2<f32>, angle: f32, fraction: f32) -> vec2<f32> {
  let unitA = geodesicUnitVector(a);
  let unitB = geodesicUnitVector(b);
  var vector: vec3<f32>;
  if (angle < 1e-6) {
    vector = normalize(mix(unitA, unitB, fraction));
  } else {
    let sinAngle = sin(angle);
    vector = unitA * (sin((1.0 - fraction) * angle) / sinAngle) +
      unitB * (sin(fraction * angle) / sinAngle);
  }
  let lngLat = geodesicFromUnitVector(vector);
  // Keep longitudes continuous with the start point (unwrapped across the antimeridian).
  return vec2<f32>(a.x + geodesicWrapLongitudeDelta(lngLat.x - a.x), lngLat.y);
}

fn geodesicSinLatitudeDelta(latitudeDegrees: f32, originLatitudeDegrees: f32) -> f32 {
  let halfSum = 0.5 * (latitudeDegrees + originLatitudeDegrees) * GEODESIC_DEGREES_TO_RADIANS;
  let halfDifference = 0.5 * (latitudeDegrees - originLatitudeDegrees) * GEODESIC_DEGREES_TO_RADIANS;
  return 2.0 * cos(halfSum) * sin(halfDifference);
}
`;
