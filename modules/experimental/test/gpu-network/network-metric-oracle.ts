// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPU_EARTH_MEAN_RADIUS_METERS} from '../../src/gpu-spatial-analysis/contracts/earth';

/** Float64 twin of the WGSL from `getNetworkMetricWGSL`. Positions are `[x, y]` or `[lng, lat]`. */
export type NetworkMetricOracle = {
  /** Euclidean distance, or the great-circle distance in meters. */
  getDistance(a: readonly number[], b: readonly number[]): number;
  /** Fraction in `[0, 1]` of the closest point of segment `start -> end` to `point`. */
  getSegmentFraction(
    point: readonly number[],
    start: readonly number[],
    end: readonly number[]
  ): number;
  /** Position at `fraction` along the segment `start -> end`. */
  interpolate(start: readonly number[], end: readonly number[], fraction: number): number[];
};

const DEGREES_TO_RADIANS = Math.PI / 180;

/** Planar metric of the network contributors. */
export const PLANAR_NETWORK_METRIC: NetworkMetricOracle = {
  getDistance: (a, b) => Math.hypot(b[0] - a[0], b[1] - a[1]),
  getSegmentFraction(point, start, end) {
    const dx = end[0] - start[0];
    const dy = end[1] - start[1];
    const denominator = dx * dx + dy * dy;
    return denominator > 0
      ? Math.min(
          1,
          Math.max(0, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / denominator)
        )
      : 0;
  },
  interpolate: (start, end, fraction) => [
    start[0] + fraction * (end[0] - start[0]),
    start[1] + fraction * (end[1] - start[1])
  ]
};

/** Wraps a longitude difference to `[-180, 180]`, as `geodesicWrapLongitudeDelta`. */
export function wrapLongitudeDelta(delta: number): number {
  return delta - 360 * Math.round(delta / 360);
}

/** Great-circle (haversine) distance in the units of `radius`. */
export function getHaversineDistance(
  a: readonly number[],
  b: readonly number[],
  radius = GPU_EARTH_MEAN_RADIUS_METERS
): number {
  const deltaPhi = (b[1] - a[1]) * DEGREES_TO_RADIANS;
  const deltaLambda = wrapLongitudeDelta(b[0] - a[0]) * DEGREES_TO_RADIANS;
  const term =
    Math.sin(deltaPhi / 2) ** 2 +
    Math.cos(a[1] * DEGREES_TO_RADIANS) *
      Math.cos(b[1] * DEGREES_TO_RADIANS) *
      Math.sin(deltaLambda / 2) ** 2;
  return 2 * radius * Math.asin(Math.sqrt(Math.min(1, term)));
}

/** Initial great-circle bearing from `a` to `b` in radians, clockwise from north. */
export function getInitialBearing(a: readonly number[], b: readonly number[]): number {
  const phi1 = a[1] * DEGREES_TO_RADIANS;
  const phi2 = b[1] * DEGREES_TO_RADIANS;
  const deltaLambda = wrapLongitudeDelta(b[0] - a[0]) * DEGREES_TO_RADIANS;
  const y = Math.sin(deltaLambda) * Math.cos(phi2);
  const x =
    Math.cos(phi1) * Math.sin(phi2) - Math.sin(phi1) * Math.cos(phi2) * Math.cos(deltaLambda);
  return Math.atan2(y, x);
}

/** Great-circle metric of the network contributors on longitude/latitude degrees. */
export function createGreatCircleNetworkMetric(
  radius = GPU_EARTH_MEAN_RADIUS_METERS
): NetworkMetricOracle {
  // Offset in the local equirectangular frame at origin, in degrees of latitude.
  const getLocalOffset = (origin: readonly number[], position: readonly number[]) => [
    wrapLongitudeDelta(position[0] - origin[0]) * Math.cos(origin[1] * DEGREES_TO_RADIANS),
    position[1] - origin[1]
  ];
  return {
    getDistance: (a, b) => getHaversineDistance(a, b, radius),
    getSegmentFraction(point, start, end) {
      const startOffset = getLocalOffset(point, start);
      const endOffset = getLocalOffset(point, end);
      const dx = endOffset[0] - startOffset[0];
      const dy = endOffset[1] - startOffset[1];
      const denominator = dx * dx + dy * dy;
      return denominator > 0
        ? Math.min(1, Math.max(0, -(startOffset[0] * dx + startOffset[1] * dy) / denominator))
        : 0;
    },
    interpolate(start, end, fraction) {
      const longitude = start[0] + fraction * wrapLongitudeDelta(end[0] - start[0]);
      return [wrapLongitudeDelta(longitude), start[1] + fraction * (end[1] - start[1])];
    }
  };
}
