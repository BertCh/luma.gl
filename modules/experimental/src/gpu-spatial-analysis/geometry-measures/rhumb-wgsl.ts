// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Largest absolute latitude, in degrees, the rhumb helpers use. The Mercator-stretched latitude
 * diverges at the poles, so inputs are clamped here (stretched latitude about 8.0). f32 spacing at
 * 90 degrees is about `7.6e-6`, so this value is representable and keeps `cos(latitude)` positive.
 */
export const GPU_RHUMB_MAX_LATITUDE = 89.9999;

/**
 * f32 rhumb-line (loxodrome) helpers on a sphere: lines of constant bearing, as in turf
 * `rhumbDistance`, `rhumbBearing`, `rhumbDestination` and the `geo` crate's `Rhumb` metric space.
 * Positions are `vec2<f32>(longitude, latitude)` in degrees. Callers prepend `GEODESIC_WGSL`.
 *
 * Precision rules:
 * - The stretched-latitude difference is `atanh((sin b - sin a) / (1 - sin a sin b))`, with the
 *   numerator from the half-angle product and the denominator written as
 *   `2 sin^2((b - a) / 2) + cos a cos b`, so short and polar edges do not cancel the two large
 *   `ln(tan(...))` terms.
 * - East-west and short-edge limit: when the latitude difference is below `1e-4 cos(mean latitude)`
 *   rad, `dPsi = dPhi / cos(mean latitude)` and the stretch ratio `dPhi / dPsi` is
 *   `cos(mean latitude)` (relative error under `1e-8`).
 * - Longitude differences are wrapped to `[-180, 180]` (the short way across the antimeridian);
 *   destinations keep longitude continuous with the origin.
 * - Latitudes are clamped to `GPU_RHUMB_MAX_LATITUDE`; a destination that overshoots a pole is
 *   reflected back across it (turf's rule) and its final bearing flips north/south.
 *
 * @internal
 */
export const RHUMB_WGSL = /* wgsl */ `
const RHUMB_MAX_LATITUDE: f32 = ${GPU_RHUMB_MAX_LATITUDE};

struct RhumbDestination {
  destination: vec2<f32>,
  finalBearing: f32,
}

fn rhumbClampLatitude(latitudeDegrees: f32) -> f32 {
  return clamp(latitudeDegrees, -RHUMB_MAX_LATITUDE, RHUMB_MAX_LATITUDE);
}

// cos of the mean latitude. Within one hemisphere the mean colatitude is formed from the exact
// colatitudes, so near a pole it keeps full relative precision (a + b would round to ~1e-5 degree).
fn rhumbMeanCosLatitude(a: f32, b: f32) -> f32 {
  if (a * b >= 0.0) {
    // colatitude(a) - (|b| - |a|) / 2, with the exact difference so nothing cancels near the pole.
    return sin(((90.0 - abs(a)) - 0.5 * (abs(b) - abs(a))) * GEODESIC_DEGREES_TO_RADIANS);
  }
  return cos(0.5 * (a + b) * GEODESIC_DEGREES_TO_RADIANS);
}

// Short edge: dPsi = dPhi / cos(mean latitude) to relative 1e-8, which keeps meter-scale edges
// exact where atanh of a ~1e-7 argument has no relative precision.
fn rhumbIsShortEdge(a: f32, b: f32) -> bool {
  let deltaPhi = (b - a) * GEODESIC_DEGREES_TO_RADIANS;
  return abs(deltaPhi) < 1e-4 * rhumbMeanCosLatitude(a, b);
}

// Stretched (isometric) latitude difference psi(b) - psi(a), with a and b in degrees.
fn rhumbStretchedDelta(a: f32, b: f32) -> f32 {
  if (rhumbIsShortEdge(a, b)) {
    return (b - a) * GEODESIC_DEGREES_TO_RADIANS / rhumbMeanCosLatitude(a, b);
  }
  let halfDelta = 0.5 * (b - a) * GEODESIC_DEGREES_TO_RADIANS;
  let sinHalfDelta = sin(halfDelta);
  let denominator = 2.0 * sinHalfDelta * sinHalfDelta +
    geodesicCosLatitude(a) * geodesicCosLatitude(b);
  // sin(b) - sin(a) = 2 cos(mean) sin(delta / 2), with the mean cosine from the exact colatitude.
  let sinDelta = 2.0 * rhumbMeanCosLatitude(a, b) * sinHalfDelta;
  return atanh(clamp(sinDelta / denominator, -0.999999, 0.999999));
}

// dPhi / dPsi, with the east-west limit cos(latitude) when the latitudes (nearly) coincide.
fn rhumbStretchRatio(a: f32, b: f32) -> f32 {
  if (rhumbIsShortEdge(a, b)) {
    return rhumbMeanCosLatitude(a, b);
  }
  return (b - a) * GEODESIC_DEGREES_TO_RADIANS / rhumbStretchedDelta(a, b);
}

// Rhumb central angle (distance / radius).
fn rhumbAngle(origin: vec2<f32>, targetPosition: vec2<f32>) -> f32 {
  let a = rhumbClampLatitude(origin.y);
  let b = rhumbClampLatitude(targetPosition.y);
  let deltaPhi = (b - a) * GEODESIC_DEGREES_TO_RADIANS;
  let deltaLambda = geodesicWrapLongitudeDelta(targetPosition.x - origin.x) * GEODESIC_DEGREES_TO_RADIANS;
  let q = rhumbStretchRatio(a, b);
  return sqrt(deltaPhi * deltaPhi + q * q * deltaLambda * deltaLambda);
}

// Constant bearing in degrees clockwise from north, (-180, 180].
fn rhumbBearingDegrees(origin: vec2<f32>, targetPosition: vec2<f32>) -> f32 {
  let a = rhumbClampLatitude(origin.y);
  let b = rhumbClampLatitude(targetPosition.y);
  let deltaLambda = geodesicWrapLongitudeDelta(targetPosition.x - origin.x) * GEODESIC_DEGREES_TO_RADIANS;
  var deltaPsi = rhumbStretchedDelta(a, b);
  // Keep atan2's second argument a positive zero (Metal mishandles -0.0).
  deltaPsi = select(deltaPsi, 0.0, deltaPsi == 0.0);
  return atan2(deltaLambda, deltaPsi) * GEODESIC_RADIANS_TO_DEGREES;
}

// Midpoint of the rhumb line: latitude and (unwrapped) longitude are both linear along it.
fn rhumbMidpoint(origin: vec2<f32>, targetPosition: vec2<f32>) -> vec2<f32> {
  let deltaLambda = geodesicWrapLongitudeDelta(targetPosition.x - origin.x);
  return vec2<f32>(origin.x + 0.5 * deltaLambda, origin.y + 0.5 * (targetPosition.y - origin.y));
}

// Destination after travelling 'angle' radians along a constant bearing.
fn rhumbDestination(origin: vec2<f32>, bearingDegrees: f32, angle: f32) -> RhumbDestination {
  let theta = bearingDegrees * GEODESIC_DEGREES_TO_RADIANS;
  var latitude = origin.y + angle * cos(theta) * GEODESIC_RADIANS_TO_DEGREES;
  var bearing = geodesicWrapLongitudeDelta(bearingDegrees);
  if (latitude > 90.0) {
    latitude = 180.0 - latitude;
    bearing = geodesicWrapLongitudeDelta(180.0 - bearing);
  } else if (latitude < -90.0) {
    latitude = -180.0 - latitude;
    bearing = geodesicWrapLongitudeDelta(180.0 - bearing);
  }
  let a = rhumbClampLatitude(origin.y);
  let b = rhumbClampLatitude(latitude);
  let q = rhumbStretchRatio(a, b);
  let deltaLambda = angle * sin(theta) / q;
  return RhumbDestination(
    vec2<f32>(origin.x + deltaLambda * GEODESIC_RADIANS_TO_DEGREES, latitude),
    bearing
  );
}
`;
