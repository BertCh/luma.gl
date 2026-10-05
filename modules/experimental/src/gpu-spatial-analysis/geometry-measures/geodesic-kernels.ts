// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWGSLFloatLiteral} from '../../utils/wgsl-kernel-nodes';
import {GPU_GEODESIC_WGS84_FLATTENING, GPU_GEODESIC_WGS84_SEMI_MAJOR_AXIS} from './geodesic-wgsl';

/** Earth model of the geodesic column contributors. */
export type GPUGeodesicModel = 'sphere' | 'wgs84';

/** Default compile-time Vincenty iteration count. */
export const GPU_GEODESIC_DEFAULT_ITERATIONS = 16;

/** Result of `vincentyInverse` in WGSL. @internal */
const VINCENTY_INVERSE_STRUCT = /* wgsl */ `
struct VincentyInverse {
  distance: f32,
  initialBearing: f32,
  finalBearing: f32,
  converged: bool,
}

struct VincentyDirect {
  destination: vec2<f32>,
  finalBearing: f32,
  converged: bool,
}
`;

/**
 * Returns the WGSL of the WGS84 Vincenty inverse and direct solutions in f32, rewritten so short
 * lines stay well conditioned: the reduced-latitude difference comes from `Δφ` minus the small
 * gap `φ - β` (never from two rounded reduced latitudes), and `cos U1 sin U2 - sin U1 cos U2 cos λ`
 * is evaluated as `sin(ΔU) + 2 sin U1 cos U2 sin²(λ / 2)`. Iterations stop at a compile-time cap;
 * rows that do not converge (near-antipodal) report `converged = false`. Callers prepend
 * `GEODESIC_WGSL`.
 *
 * @internal
 */
export function getVincentyWGSL(iterations: number): string {
  const semiMinorAxis = GPU_GEODESIC_WGS84_SEMI_MAJOR_AXIS * (1 - GPU_GEODESIC_WGS84_FLATTENING);
  const a = GPU_GEODESIC_WGS84_SEMI_MAJOR_AXIS;
  return /* wgsl */ `
${VINCENTY_INVERSE_STRUCT}
const VINCENTY_A: f32 = ${getWGSLFloatLiteral(a)};
const VINCENTY_B: f32 = ${getWGSLFloatLiteral(semiMinorAxis)};
const VINCENTY_F: f32 = ${getWGSLFloatLiteral(GPU_GEODESIC_WGS84_FLATTENING)};
// (a^2 - b^2) / b^2
const VINCENTY_SECOND_ECCENTRICITY_SQUARED: f32 = ${getWGSLFloatLiteral((a * a - semiMinorAxis * semiMinorAxis) / (semiMinorAxis * semiMinorAxis))};
const VINCENTY_ITERATIONS: u32 = ${iterations}u;

fn getReducedLatitudeGap(phi: f32) -> f32 {
  let sinPhi = sin(phi);
  let cosPhi = cos(phi);
  return atan(VINCENTY_F * sinPhi * cosPhi / (cosPhi * cosPhi + (1.0 - VINCENTY_F) * sinPhi * sinPhi));
}

// cos(U) as sin(colatitude of U) = sin((90 - |lat|) degrees + gap(|phi|)), exact near the poles.
fn getCosReducedLatitude(phi: f32, latitudeDegrees: f32) -> f32 {
  return sin((90.0 - abs(latitudeDegrees)) * GEODESIC_DEGREES_TO_RADIANS + getReducedLatitudeGap(abs(phi)));
}

fn getReducedLatitude(phi: f32) -> f32 {
  return phi - getReducedLatitudeGap(phi);
}

fn getVincentyDeltaSigma(bigB: f32, sinSigma: f32, cosSigma: f32, cos2SigmaM: f32) -> f32 {
  let cos2SigmaMSquared = cos2SigmaM * cos2SigmaM;
  return bigB * sinSigma * (cos2SigmaM + bigB / 4.0 * (cosSigma * (-1.0 + 2.0 * cos2SigmaMSquared) -
    bigB / 6.0 * cos2SigmaM * (-3.0 + 4.0 * sinSigma * sinSigma) * (-3.0 + 4.0 * cos2SigmaMSquared)));
}

fn getVincentyCoefficients(cosSquaredAlpha: f32) -> vec2<f32> {
  let uSquared = cosSquaredAlpha * VINCENTY_SECOND_ECCENTRICITY_SQUARED;
  let bigA = 1.0 + uSquared / 16384.0 * (4096.0 + uSquared * (-768.0 + uSquared * (320.0 - 175.0 * uSquared)));
  let bigB = uSquared / 1024.0 * (256.0 + uSquared * (-128.0 + uSquared * (74.0 - 47.0 * uSquared)));
  return vec2<f32>(bigA, bigB);
}

fn vincentyInverse(a: vec2<f32>, b: vec2<f32>) -> VincentyInverse {
  let phi1 = a.y * GEODESIC_DEGREES_TO_RADIANS;
  let phi2 = b.y * GEODESIC_DEGREES_TO_RADIANS;
  let deltaPhi = (b.y - a.y) * GEODESIC_DEGREES_TO_RADIANS;
  let reduced1 = getReducedLatitude(phi1);
  let reduced2 = getReducedLatitude(phi2);
  // U2 - U1: the midpoint derivative dU/dphi for short spans (the gap difference would lose
  // relative precision there), else deltaPhi minus the difference of the small gaps.
  var deltaReduced = deltaPhi - (getReducedLatitudeGap(phi2) - getReducedLatitudeGap(phi1));
  if (abs(deltaPhi) < 0.01) {
    let sinMid = sin(0.5 * (phi1 + phi2));
    let cosMid = cos(0.5 * (phi1 + phi2));
    deltaReduced = deltaPhi * (1.0 - VINCENTY_F) /
      (cosMid * cosMid + (1.0 - VINCENTY_F) * (1.0 - VINCENTY_F) * sinMid * sinMid);
  }
  let sinU1 = sin(reduced1);
  let cosU1 = getCosReducedLatitude(phi1, a.y);
  let sinU2 = sin(reduced2);
  let cosU2 = getCosReducedLatitude(phi2, b.y);
  let sinDeltaU = sin(deltaReduced);
  let longitudeDelta = geodesicWrapLongitudeDelta(b.x - a.x) * GEODESIC_DEGREES_TO_RADIANS;
  var lambda = longitudeDelta;
  var result: VincentyInverse;
  result.converged = false;
  var sinSigma = 0.0;
  var cosSigma = 1.0;
  var sigma = 0.0;
  var cosSquaredAlpha = 1.0;
  var cos2SigmaM = 0.0;
  var sinLambda = 0.0;
  var halfVersine = 0.0;
  // Each pass evaluates the auxiliary sphere at lambda; the final pass after convergence (or the
  // cap) evaluates it at the converged lambda, so the distance never mixes two lambda values.
  for (var iteration = 0u; iteration <= VINCENTY_ITERATIONS; iteration++) {
    sinLambda = sin(lambda);
    let sinHalfLambda = sin(0.5 * lambda);
    halfVersine = 2.0 * sinHalfLambda * sinHalfLambda;
    let northTerm = sinDeltaU + sinU1 * cosU2 * halfVersine;
    let eastTerm = cosU2 * sinLambda;
    sinSigma = sqrt(eastTerm * eastTerm + northTerm * northTerm);
    if (sinSigma == 0.0) {
      result.distance = 0.0;
      result.initialBearing = 0.0;
      result.finalBearing = 0.0;
      result.converged = true;
      return result;
    }
    cosSigma = sinU1 * sinU2 + cosU1 * cosU2 * (1.0 - halfVersine);
    sigma = atan2(sinSigma, cosSigma);
    let sinAlpha = cosU1 * cosU2 * sinLambda / sinSigma;
    cosSquaredAlpha = 1.0 - sinAlpha * sinAlpha;
    cos2SigmaM = 0.0;
    if (cosSquaredAlpha != 0.0) {
      cos2SigmaM = cosSigma - 2.0 * sinU1 * sinU2 / cosSquaredAlpha;
    }
    if (result.converged || iteration == VINCENTY_ITERATIONS) {
      break;
    }
    let c = VINCENTY_F / 16.0 * cosSquaredAlpha * (4.0 + VINCENTY_F * (4.0 - 3.0 * cosSquaredAlpha));
    let previous = lambda;
    lambda = longitudeDelta + (1.0 - c) * VINCENTY_F * sinAlpha *
      (sigma + c * sinSigma * (cos2SigmaM + c * cosSigma * (-1.0 + 2.0 * cos2SigmaM * cos2SigmaM)));
    // Relative test: about four f32 ulps of lambda (exact zero for meridian lines).
    if (abs(lambda - previous) <= 4.8e-7 * abs(lambda)) {
      result.converged = true;
    }
  }
  let coefficients = getVincentyCoefficients(cosSquaredAlpha);
  let deltaSigma = getVincentyDeltaSigma(coefficients.y, sinSigma, cosSigma, cos2SigmaM);
  result.distance = VINCENTY_B * coefficients.x * (sigma - deltaSigma);
  result.initialBearing = atan2(cosU2 * sinLambda, sinDeltaU + sinU1 * cosU2 * halfVersine) *
    GEODESIC_RADIANS_TO_DEGREES;
  result.finalBearing = atan2(cosU1 * sinLambda, sinDeltaU - cosU1 * sinU2 * halfVersine) *
    GEODESIC_RADIANS_TO_DEGREES;
  return result;
}

fn vincentyDirect(origin: vec2<f32>, bearingDegrees: f32, distance: f32) -> VincentyDirect {
  let phi1 = origin.y * GEODESIC_DEGREES_TO_RADIANS;
  let alpha1 = bearingDegrees * GEODESIC_DEGREES_TO_RADIANS;
  let sinAlpha1 = sin(alpha1);
  let cosAlpha1 = cos(alpha1);
  let reduced1 = getReducedLatitude(phi1);
  let sinU1 = sin(reduced1);
  let cosU1 = getCosReducedLatitude(phi1, origin.y);
  let sigma1 = atan2(sinU1, cosU1 * cosAlpha1);
  let sinAlpha = cosU1 * sinAlpha1;
  let cosSquaredAlpha = 1.0 - sinAlpha * sinAlpha;
  let coefficients = getVincentyCoefficients(cosSquaredAlpha);
  let baseSigma = distance / (VINCENTY_B * coefficients.x);
  var sigma = baseSigma;
  var sinSigma = sin(sigma);
  var cosSigma = cos(sigma);
  var cos2SigmaM = cos(2.0 * sigma1 + sigma);
  var result: VincentyDirect;
  result.converged = false;
  for (var iteration = 0u; iteration < VINCENTY_ITERATIONS; iteration++) {
    cos2SigmaM = cos(2.0 * sigma1 + sigma);
    sinSigma = sin(sigma);
    cosSigma = cos(sigma);
    let previous = sigma;
    sigma = baseSigma + getVincentyDeltaSigma(coefficients.y, sinSigma, cosSigma, cos2SigmaM);
    if (abs(sigma - previous) <= 2.4e-7 * max(1.0, sigma)) {
      result.converged = true;
      break;
    }
  }
  sinSigma = sin(sigma);
  cosSigma = cos(sigma);
  cos2SigmaM = cos(2.0 * sigma1 + sigma);
  let northTerm = sinU1 * sinSigma - cosU1 * cosSigma * cosAlpha1;
  let phi2 = atan2(
    sinU1 * cosSigma + cosU1 * sinSigma * cosAlpha1,
    (1.0 - VINCENTY_F) * sqrt(sinAlpha * sinAlpha + northTerm * northTerm)
  );
  let lambda = atan2(sinSigma * sinAlpha1, cosU1 * cosSigma - sinU1 * sinSigma * cosAlpha1);
  let c = VINCENTY_F / 16.0 * cosSquaredAlpha * (4.0 + VINCENTY_F * (4.0 - 3.0 * cosSquaredAlpha));
  let longitudeDelta = lambda - (1.0 - c) * VINCENTY_F * sinAlpha *
    (sigma + c * sinSigma * (cos2SigmaM + c * cosSigma * (-1.0 + 2.0 * cos2SigmaM * cos2SigmaM)));
  result.destination = vec2<f32>(
    origin.x + longitudeDelta * GEODESIC_RADIANS_TO_DEGREES,
    phi2 * GEODESIC_RADIANS_TO_DEGREES
  );
  result.finalBearing = atan2(sinAlpha, -northTerm) * GEODESIC_RADIANS_TO_DEGREES;
  return result;
}
`;
}

/** Sphere final bearing: the reverse initial bearing turned by 180 degrees. @internal */
export const SPHERE_PAIR_WGSL = /* wgsl */ `
fn sphereFinalBearingDegrees(a: vec2<f32>, b: vec2<f32>) -> f32 {
  var bearing = geodesicInitialBearingDegrees(b, a) + 180.0;
  if (bearing > 180.0) {
    bearing -= 360.0;
  }
  return bearing;
}
`;
