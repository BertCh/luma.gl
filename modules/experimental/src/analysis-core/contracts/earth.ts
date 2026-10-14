// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * IUGG mean Earth radius R1 = (2a + b) / 3 of WGS84, in meters. The default sphere for
 * great-circle distances, bearings and destinations.
 */
export const GPU_EARTH_MEAN_RADIUS_METERS = 6371008.8;

/** WGS84 authalic (equal-area) sphere radius in meters, used for spherical areas. */
export const GPU_EARTH_AUTHALIC_RADIUS_METERS = 6371007.180918475;

/** WGS84 semi-major axis in meters. */
export const GPU_WGS84_SEMI_MAJOR_AXIS_METERS = 6378137;

/** WGS84 inverse flattening. */
export const GPU_WGS84_INVERSE_FLATTENING = 298.257223563;

/** WGS84 flattening. */
export const GPU_WGS84_FLATTENING = 1 / GPU_WGS84_INVERSE_FLATTENING;

/**
 * cuSpatial parity — cuSpatial's sinusoidal projection assumes a 40,000 km circumference
 * (111.111 km per degree), not the mean-sphere 111.195 km; kept only for
 * `GPUSinusoidalProjection` parity.
 */
export const GPU_SINUSOIDAL_KILOMETERS_PER_DEGREE = 40_000 / 360;
