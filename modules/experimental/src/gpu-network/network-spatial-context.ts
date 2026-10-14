// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPU_EARTH_MEAN_RADIUS_METERS} from '../gpu-spatial-analysis/contracts/earth';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext,
  type GPUSpatialContextSupport
} from '../gpu-spatial-analysis/contracts/spatial-context';
import {GEODESIC_WGSL} from '../gpu-spatial-analysis/geometry-measures/geodesic-wgsl';

/**
 * Spatial contexts accepted by the network contributors that have a great-circle path: native
 * planar coordinates, or `{coordinateSpace: 'longitude-latitude', metric: 'great-circle'}`.
 *
 * @internal
 */
export const GPU_NETWORK_GEODESIC_SPATIAL_CONTEXT_SUPPORT: GPUSpatialContextSupport = Object.freeze(
  {
    coordinateSpaces: Object.freeze(['planar', 'longitude-latitude'] as const),
    metrics: Object.freeze(['none', 'native', 'great-circle'] as const)
  }
);

/**
 * Resolves the context of a network contributor with a great-circle path. Longitude/latitude
 * positions need an explicit `'great-circle'` metric, and great-circle results are in meters.
 *
 * @internal
 */
export function resolveNetworkSpatialContext(
  id: string,
  context: GPUSpatialContext | undefined
): GPUSpatialContext {
  const resolved = resolveGPUSpatialContext(
    id,
    context,
    GPU_NETWORK_GEODESIC_SPATIAL_CONTEXT_SUPPORT
  );
  if (resolved.coordinateSpace === 'longitude-latitude' && resolved.metric !== 'great-circle') {
    throw new Error(`${id} longitude-latitude positions need the great-circle metric`);
  }
  if (resolved.units === 'kilometers') {
    throw new Error(`${id} great-circle results are in meters`);
  }
  return resolved;
}

/**
 * Sphere radius in meters of a great-circle context, or `undefined` for planar coordinates.
 *
 * @internal
 */
export function getNetworkGeodesicRadius(context: GPUSpatialContext): number | undefined {
  return context.metric === 'great-circle'
    ? (context.sphereRadius ?? GPU_EARTH_MEAN_RADIUS_METERS)
    : undefined;
}

/**
 * WGSL point and segment metric shared by the network contributors. Include it once per shader.
 *
 * - `networkDistance(a, b)`: Euclidean distance, or the great-circle distance in meters.
 * - `networkSegmentFraction(point, start, end)`: fraction in `[0, 1]` of the point of the segment
 *   closest to `point`. On longitude/latitude it is computed in the local equirectangular frame at
 *   `point` (east offsets scaled by `cos(latitude)`, longitudes wrapped at +-180), which is exact to
 *   first order for road-length segments.
 * - `networkInterpolate(start, end, fraction)`: the position at `fraction` along the segment, the
 *   straight segment in the coordinates (the shorter way around in longitude, wrapped to +-180).
 * - `networkSegmentDistance(point, start, end, fraction)`: distance from `point` to the position at
 *   `fraction`, without rounding that position to absolute coordinates first (an f32 longitude
 *   near +-180 is only good to about a meter), so short snap distances keep their precision.
 *
 * The great-circle variants wrap the public `GEODESIC_WGSL` helpers (`geodesicHaversineDistance`,
 * `geodesicSegmentFraction`, `geodesicSegmentDistance`); only `networkInterpolate` is local, since
 * `geodesicInterpolate` follows the great circle while network edges are straight in degrees.
 *
 * Network edges are straight segments between node positions, so on longitude/latitude they are
 * meant to be road-length: edge lengths are great-circle chords of their end points.
 *
 * @internal
 */
export function getNetworkMetricWGSL(radius: number | undefined): string {
  if (radius === undefined) {
    return /* wgsl */ `
fn networkDistance(a: vec2<f32>, b: vec2<f32>) -> f32 {
  let delta = a - b;
  return sqrt(dot(delta, delta));
}
fn networkSegmentFraction(point: vec2<f32>, start: vec2<f32>, end: vec2<f32>) -> f32 {
  let segment = end - start;
  let denominator = dot(segment, segment);
  if (denominator > 0.0) {
    return clamp(dot(point - start, segment) / denominator, 0.0, 1.0);
  }
  return 0.0;
}
fn networkInterpolate(start: vec2<f32>, end: vec2<f32>, fraction: f32) -> vec2<f32> {
  return start + fraction * (end - start);
}
fn networkSegmentDistance(point: vec2<f32>, start: vec2<f32>, end: vec2<f32>, fraction: f32) -> f32 {
  let offset = point - (start + fraction * (end - start));
  return sqrt(dot(offset, offset));
}`;
  }
  return /* wgsl */ `${GEODESIC_WGSL}
const NETWORK_EARTH_RADIUS: f32 = ${toWGSLFloat(radius)};
fn networkDistance(a: vec2<f32>, b: vec2<f32>) -> f32 {
  return geodesicHaversineDistance(a, b, NETWORK_EARTH_RADIUS);
}
fn networkSegmentFraction(point: vec2<f32>, start: vec2<f32>, end: vec2<f32>) -> f32 {
  return geodesicSegmentFraction(point, start, end);
}
fn networkInterpolate(start: vec2<f32>, end: vec2<f32>, fraction: f32) -> vec2<f32> {
  let delta = vec2<f32>(geodesicWrapLongitudeDelta(end.x - start.x), end.y - start.y);
  let position = start + fraction * delta;
  return vec2<f32>(geodesicWrapLongitudeDelta(position.x), position.y);
}
fn networkSegmentDistance(point: vec2<f32>, start: vec2<f32>, end: vec2<f32>, fraction: f32) -> f32 {
  return geodesicSegmentDistance(point, start, end, fraction, NETWORK_EARTH_RADIUS);
}`;
}

function toWGSLFloat(value: number): string {
  const text = String(Math.fround(value));
  return /[.eE]/.test(text) ? text : `${text}.0`;
}
