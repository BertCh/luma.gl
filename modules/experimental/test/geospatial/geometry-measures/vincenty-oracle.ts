// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {wrapLongitudeDelta} from './geodesic-oracle';

const DEGREES_TO_RADIANS = Math.PI / 180;
const RADIANS_TO_DEGREES = 180 / Math.PI;
const A = 6378137;
const F = 1 / 298.257223563;
const B = A * (1 - F);

function getCoefficients(cosSquaredAlpha: number): {bigA: number; bigB: number} {
  const uSquared = (cosSquaredAlpha * (A * A - B * B)) / (B * B);
  return {
    bigA: 1 + (uSquared / 16384) * (4096 + uSquared * (-768 + uSquared * (320 - 175 * uSquared))),
    bigB: (uSquared / 1024) * (256 + uSquared * (-128 + uSquared * (74 - 47 * uSquared)))
  };
}

function getDeltaSigma(
  bigB: number,
  sinSigma: number,
  cosSigma: number,
  cos2SigmaM: number
): number {
  return (
    bigB *
    sinSigma *
    (cos2SigmaM +
      (bigB / 4) *
        (cosSigma * (-1 + 2 * cos2SigmaM ** 2) -
          (bigB / 6) * cos2SigmaM * (-3 + 4 * sinSigma ** 2) * (-3 + 4 * cos2SigmaM ** 2)))
  );
}

/** f64 Vincenty inverse on WGS84: distance in meters and bearings in `(-180, 180]` degrees. */
export function getVincentyInverse(
  a: readonly number[],
  b: readonly number[]
): {distance: number; initialBearing: number; finalBearing: number; converged: boolean} {
  const longitudeDelta = wrapLongitudeDelta(b[0] - a[0]) * DEGREES_TO_RADIANS;
  const u1 = Math.atan((1 - F) * Math.tan(a[1] * DEGREES_TO_RADIANS));
  const u2 = Math.atan((1 - F) * Math.tan(b[1] * DEGREES_TO_RADIANS));
  const [sinU1, cosU1, sinU2, cosU2] = [Math.sin(u1), Math.cos(u1), Math.sin(u2), Math.cos(u2)];
  let lambda = longitudeDelta;
  for (let iteration = 0; iteration < 1000; iteration++) {
    const sinLambda = Math.sin(lambda);
    const cosLambda = Math.cos(lambda);
    const sinSigma = Math.hypot(cosU2 * sinLambda, cosU1 * sinU2 - sinU1 * cosU2 * cosLambda);
    if (sinSigma === 0) {
      return {distance: 0, initialBearing: 0, finalBearing: 0, converged: true};
    }
    const cosSigma = sinU1 * sinU2 + cosU1 * cosU2 * cosLambda;
    const sigma = Math.atan2(sinSigma, cosSigma);
    const sinAlpha = (cosU1 * cosU2 * sinLambda) / sinSigma;
    const cosSquaredAlpha = 1 - sinAlpha * sinAlpha;
    const cos2SigmaM = cosSquaredAlpha === 0 ? 0 : cosSigma - (2 * sinU1 * sinU2) / cosSquaredAlpha;
    const c = (F / 16) * cosSquaredAlpha * (4 + F * (4 - 3 * cosSquaredAlpha));
    const previous = lambda;
    lambda =
      longitudeDelta +
      (1 - c) *
        F *
        sinAlpha *
        (sigma + c * sinSigma * (cos2SigmaM + c * cosSigma * (-1 + 2 * cos2SigmaM ** 2)));
    if (Math.abs(lambda - previous) < 1e-13) {
      const {bigA, bigB} = getCoefficients(cosSquaredAlpha);
      const finalSinLambda = Math.sin(lambda);
      const finalCosLambda = Math.cos(lambda);
      return {
        distance: B * bigA * (sigma - getDeltaSigma(bigB, sinSigma, cosSigma, cos2SigmaM)),
        initialBearing:
          Math.atan2(cosU2 * finalSinLambda, cosU1 * sinU2 - sinU1 * cosU2 * finalCosLambda) *
          RADIANS_TO_DEGREES,
        finalBearing:
          Math.atan2(cosU1 * finalSinLambda, -sinU1 * cosU2 + cosU1 * sinU2 * finalCosLambda) *
          RADIANS_TO_DEGREES,
        converged: true
      };
    }
  }
  return {distance: NaN, initialBearing: NaN, finalBearing: NaN, converged: false};
}

/** f64 Vincenty direct on WGS84: destination (longitude continuous with the origin) and final bearing. */
export function getVincentyDirect(
  origin: readonly number[],
  bearingDegrees: number,
  distance: number
): {destination: [number, number]; finalBearing: number} {
  const alpha1 = bearingDegrees * DEGREES_TO_RADIANS;
  const [sinAlpha1, cosAlpha1] = [Math.sin(alpha1), Math.cos(alpha1)];
  const u1 = Math.atan((1 - F) * Math.tan(origin[1] * DEGREES_TO_RADIANS));
  const [sinU1, cosU1] = [Math.sin(u1), Math.cos(u1)];
  const sigma1 = Math.atan2(sinU1, cosU1 * cosAlpha1);
  const sinAlpha = cosU1 * sinAlpha1;
  const cosSquaredAlpha = 1 - sinAlpha * sinAlpha;
  const {bigA, bigB} = getCoefficients(cosSquaredAlpha);
  const baseSigma = distance / (B * bigA);
  let sigma = baseSigma;
  for (let iteration = 0; iteration < 1000; iteration++) {
    const cos2SigmaM = Math.cos(2 * sigma1 + sigma);
    const previous = sigma;
    sigma = baseSigma + getDeltaSigma(bigB, Math.sin(sigma), Math.cos(sigma), cos2SigmaM);
    if (Math.abs(sigma - previous) < 1e-14) {
      break;
    }
  }
  const [sinSigma, cosSigma] = [Math.sin(sigma), Math.cos(sigma)];
  const cos2SigmaM = Math.cos(2 * sigma1 + sigma);
  const northTerm = sinU1 * sinSigma - cosU1 * cosSigma * cosAlpha1;
  const phi2 = Math.atan2(
    sinU1 * cosSigma + cosU1 * sinSigma * cosAlpha1,
    (1 - F) * Math.hypot(sinAlpha, northTerm)
  );
  const lambda = Math.atan2(sinSigma * sinAlpha1, cosU1 * cosSigma - sinU1 * sinSigma * cosAlpha1);
  const c = (F / 16) * cosSquaredAlpha * (4 + F * (4 - 3 * cosSquaredAlpha));
  const longitudeDelta =
    lambda -
    (1 - c) *
      F *
      sinAlpha *
      (sigma + c * sinSigma * (cos2SigmaM + c * cosSigma * (-1 + 2 * cos2SigmaM ** 2)));
  return {
    destination: [origin[0] + longitudeDelta * RADIANS_TO_DEGREES, phi2 * RADIANS_TO_DEGREES],
    finalBearing: Math.atan2(sinAlpha, -northTerm) * RADIANS_TO_DEGREES
  };
}
