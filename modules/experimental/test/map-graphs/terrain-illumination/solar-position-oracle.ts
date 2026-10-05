// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

const RADIANS = Math.PI / 180;

/**
 * Independent sun position from the Astronomical Almanac low-precision formulas (right ascension
 * and Greenwich mean sidereal time instead of the equation of time; about 0.01 degree for
 * 1950-2050).
 *
 * Returns geometric altitude and azimuth clockwise from north in degrees.
 */
export function getAlmanacSolarPosition(
  timestamp: number,
  longitude: number,
  latitude: number
): {
  azimuthDegrees: number;
  altitudeDegrees: number;
  declinationDegrees: number;
} {
  const days = (timestamp - Date.UTC(2000, 0, 1, 12)) / 86400000;
  const meanLongitude = 280.46 + 0.9856474 * days;
  const meanAnomaly = (357.528 + 0.9856003 * days) * RADIANS;
  const eclipticLongitude =
    (meanLongitude + 1.915 * Math.sin(meanAnomaly) + 0.02 * Math.sin(2 * meanAnomaly)) * RADIANS;
  const obliquity = (23.439 - 0.0000004 * days) * RADIANS;
  const rightAscension = Math.atan2(
    Math.cos(obliquity) * Math.sin(eclipticLongitude),
    Math.cos(eclipticLongitude)
  );
  const declination = Math.asin(Math.sin(obliquity) * Math.sin(eclipticLongitude));
  const siderealHours = 18.697374558 + 24.06570982441908 * days;
  const hourAngle = (siderealHours * 15 + longitude) * RADIANS - rightAscension;
  const phi = latitude * RADIANS;
  const sinAltitude =
    Math.sin(phi) * Math.sin(declination) +
    Math.cos(phi) * Math.cos(declination) * Math.cos(hourAngle);
  const altitude = Math.asin(Math.min(Math.max(sinAltitude, -1), 1));
  // Direction vector in east/north components, then azimuth from north.
  const east = -Math.cos(declination) * Math.sin(hourAngle);
  const north =
    Math.cos(phi) * Math.sin(declination) -
    Math.sin(phi) * Math.cos(declination) * Math.cos(hourAngle);
  return {
    azimuthDegrees: (((Math.atan2(east, north) / RADIANS) % 360) + 360) % 360,
    altitudeDegrees: altitude / RADIANS,
    declinationDegrees: declination / RADIANS
  };
}

/** Smallest absolute difference between two angles in degrees. */
export function getAngleDifference(left: number, right: number): number {
  const difference = Math.abs(left - right) % 360;
  return Math.min(difference, 360 - difference);
}
