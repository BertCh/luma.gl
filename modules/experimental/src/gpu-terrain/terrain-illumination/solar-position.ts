// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

const MILLISECONDS_PER_DAY = 86400000;
/** Unix epoch milliseconds of J2000.0, 2000-01-01T12:00:00Z. */
const J2000_EPOCH_MILLISECONDS = 946728000000;
const DEGREES = 180 / Math.PI;
const RADIANS = Math.PI / 180;

/** Sun position returned by {@link getSolarPosition}. */
export type SolarPosition = {
  /** Direction of the sun, degrees clockwise from north in `[0, 360)`. */
  azimuthDegrees: number;
  /** Elevation of the sun center above the horizon in degrees, refracted unless disabled. */
  altitudeDegrees: number;
  /** Geometric (unrefracted) elevation of the sun center in degrees. */
  geometricAltitudeDegrees: number;
  /** `90 - altitudeDegrees`. */
  zenithDegrees: number;
  /** Solar declination in degrees. */
  declinationDegrees: number;
  /** Equation of time in minutes (apparent minus mean solar time). */
  equationOfTimeMinutes: number;
  /** Local hour angle in degrees, negative before solar noon, in `[-180, 180)`. */
  hourAngleDegrees: number;
};

/** Options for {@link getSolarPosition}. */
export type SolarPositionOptions = {
  /** Apply the NOAA atmospheric refraction approximation to `altitudeDegrees`. Defaults to true. */
  refraction?: boolean;
};

/** Days since J2000.0 split for float32 transport, see {@link getSolarTimeParameter}. */
export type SolarTimeParameter = {
  /** Whole days since 2000-01-01T12:00:00Z (exact in float32 for +-45,000 years). */
  dayNumber: number;
  /** Fraction of the day in `[0, 1)` after `dayNumber`, counted from 12:00 UTC. */
  dayFraction: number;
};

/**
 * Splits a timestamp into whole days since J2000.0 and a day fraction.
 *
 * Both values survive float32 storage with sub-second precision, unlike a single day count, so
 * GPU kernels can rebuild the hour angle exactly.
 */
export function getSolarTimeParameter(timestamp: number | Date): SolarTimeParameter {
  const milliseconds = typeof timestamp === 'number' ? timestamp : timestamp.getTime();
  if (!Number.isFinite(milliseconds)) {
    throw new Error('Solar time must be a finite timestamp');
  }
  const days = (milliseconds - J2000_EPOCH_MILLISECONDS) / MILLISECONDS_PER_DAY;
  const dayNumber = Math.floor(days);
  return {dayNumber, dayFraction: days - dayNumber};
}

/**
 * Returns the NOAA atmospheric refraction correction in degrees for a geometric elevation.
 *
 * Piecewise approximation from the NOAA solar calculator: 0 above 85 degrees, a cotangent series
 * above 5 degrees, a polynomial down to -0.575 degrees, and `-20.772 / tan(e)` below.
 */
export function getSolarRefractionDegrees(elevationDegrees: number): number {
  if (elevationDegrees > 85) {
    return 0;
  }
  const tangent = Math.tan(elevationDegrees * RADIANS);
  let arcseconds: number;
  if (elevationDegrees > 5) {
    arcseconds = 58.1 / tangent - 0.07 / tangent ** 3 + 0.000086 / tangent ** 5;
  } else if (elevationDegrees > -0.575) {
    const e = elevationDegrees;
    arcseconds = 1735 + e * (-518.2 + e * (103.4 + e * (-12.79 + e * 0.711)));
  } else {
    arcseconds = -20.772 / tangent;
  }
  return arcseconds / 3600;
}

/**
 * Computes the sun position with the NOAA solar calculator algorithm (Meeus, low precision).
 *
 * Accuracy is about 0.01 degree for years 1800-2100 before refraction. The computation runs in
 * float64 on the CPU; pass `azimuthDegrees` and `altitudeDegrees` to per-frame recipe settings
 * such as `getGPUSolarShadowMaskParameterValues`, or use `GPUSolarPosition` for many locations.
 *
 * @param timestamp Unix epoch milliseconds (UTC) or a `Date`.
 * @param longitude Degrees east.
 * @param latitude Degrees north in `[-90, 90]`.
 */
export function getSolarPosition(
  timestamp: number | Date,
  longitude: number,
  latitude: number,
  options: SolarPositionOptions = {}
): SolarPosition {
  const {dayNumber, dayFraction} = getSolarTimeParameter(timestamp);
  const julianCentury = (dayNumber + dayFraction) / 36525;
  const meanLongitude = modulo(
    280.46646 + julianCentury * (36000.76983 + julianCentury * 0.0003032),
    360
  );
  const meanAnomaly = 357.52911 + julianCentury * (35999.05029 - 0.0001537 * julianCentury);
  const eccentricity = 0.016708634 - julianCentury * (0.000042037 + 0.0000001267 * julianCentury);
  const anomaly = meanAnomaly * RADIANS;
  const equationOfCenter =
    Math.sin(anomaly) * (1.914602 - julianCentury * (0.004817 + 0.000014 * julianCentury)) +
    Math.sin(2 * anomaly) * (0.019993 - 0.000101 * julianCentury) +
    Math.sin(3 * anomaly) * 0.000289;
  const omega = (125.04 - 1934.136 * julianCentury) * RADIANS;
  const apparentLongitude =
    (meanLongitude + equationOfCenter - 0.00569 - 0.00478 * Math.sin(omega)) * RADIANS;
  const meanObliquity =
    23 +
    (26 +
      (21.448 - julianCentury * (46.815 + julianCentury * (0.00059 - julianCentury * 0.001813))) /
        60) /
      60;
  const obliquity = (meanObliquity + 0.00256 * Math.cos(omega)) * RADIANS;
  const declination = Math.asin(Math.sin(obliquity) * Math.sin(apparentLongitude));
  const y = Math.tan(obliquity / 2) ** 2;
  const longitudeRadians = meanLongitude * RADIANS;
  const equationOfTimeMinutes =
    4 *
    DEGREES *
    (y * Math.sin(2 * longitudeRadians) -
      2 * eccentricity * Math.sin(anomaly) +
      4 * eccentricity * y * Math.sin(anomaly) * Math.cos(2 * longitudeRadians) -
      0.5 * y * y * Math.sin(4 * longitudeRadians) -
      1.25 * eccentricity * eccentricity * Math.sin(2 * anomaly));
  // J2000 days start at 12:00 UTC.
  const minutesUTC = modulo(dayFraction + 0.5, 1) * 1440;
  const trueSolarTime = modulo(minutesUTC + equationOfTimeMinutes + 4 * longitude, 1440);
  const hourAngleDegrees =
    trueSolarTime / 4 < 0 ? trueSolarTime / 4 + 180 : trueSolarTime / 4 - 180;
  const latitudeRadians = latitude * RADIANS;
  const hourAngle = hourAngleDegrees * RADIANS;
  const cosZenith = clamp(
    Math.sin(latitudeRadians) * Math.sin(declination) +
      Math.cos(latitudeRadians) * Math.cos(declination) * Math.cos(hourAngle),
    -1,
    1
  );
  const zenith = Math.acos(cosZenith);
  const geometricAltitudeDegrees = 90 - zenith * DEGREES;
  const altitudeDegrees =
    options.refraction === false
      ? geometricAltitudeDegrees
      : geometricAltitudeDegrees + getSolarRefractionDegrees(geometricAltitudeDegrees);
  // Azimuth from north via atan2 (equivalent to the NOAA acos form, well conditioned at the poles).
  const azimuthDegrees = modulo(
    Math.atan2(
      Math.sin(hourAngle),
      Math.cos(hourAngle) * Math.sin(latitudeRadians) -
        Math.tan(declination) * Math.cos(latitudeRadians)
    ) *
      DEGREES +
      180,
    360
  );
  return {
    azimuthDegrees,
    altitudeDegrees,
    geometricAltitudeDegrees,
    zenithDegrees: 90 - altitudeDegrees,
    declinationDegrees: declination * DEGREES,
    equationOfTimeMinutes,
    hourAngleDegrees
  };
}

function modulo(value: number, divisor: number): number {
  return value - divisor * Math.floor(value / divisor);
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(Math.max(value, minimum), maximum);
}
