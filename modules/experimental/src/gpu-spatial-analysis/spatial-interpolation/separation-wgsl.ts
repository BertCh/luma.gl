// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getGPUSphereRadius,
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {getWGSLFloatLiteral} from '../../utils/wgsl-kernel-nodes';
import {GEODESIC_WGSL} from '../geometry-measures/geodesic-wgsl';

/** Resolved metric of an interpolation operator. @internal */
export type InterpolationSeparation = {
  context: GPUSpatialContext;
  /** True for longitude/latitude positions with great-circle distances. */
  geographic: boolean;
  /** Sphere radius in the distance unit of the context (meters, or kilometers). */
  sphereRadius: number;
};

/**
 * Resolves a spatial context for an operator that supports planar coordinates and
 * longitude/latitude with the `'great-circle'` metric (haversine on a sphere). Any other
 * longitude/latitude metric throws.
 * @internal
 */
export function resolveInterpolationSeparation(
  id: string,
  context: GPUSpatialContext | undefined
): InterpolationSeparation {
  const resolved = resolveGPUSpatialContext(
    id,
    context,
    GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT
  );
  const geographic = resolved.coordinateSpace === 'longitude-latitude';
  return {context: resolved, geographic, sphereRadius: getGPUSphereRadius(resolved)};
}

/**
 * WGSL for the separation between a sample and a query point, shared by the interpolation kernels.
 *
 * Functions: `getSeparationKey(position, query)` (an order-preserving key: squared distance in the
 * planar case, the haversine term in the great-circle case), `getRadiusKey(radius)`,
 * `getSeparationMeters(key)` and `getSquaredMeters(key)` (distance and squared distance in the
 * context unit), `getPairMeters(a, b)`, `isExactHit(key, position, query)` and
 * `getQueryBox(center, radius)` (a conservative box in index coordinates, degrees for
 * great-circle). Great-circle positions are absolute lon/lat, so the local-origin offsets are zero.
 * @internal
 */
export function getSeparationWGSL(separation: InterpolationSeparation): string {
  if (!separation.geographic) {
    return /* wgsl */ `
struct QueryBox { minimum: vec2f, maximum: vec2f, }
fn getSeparationKey(position: vec2f, query: vec2f) -> f32 {
  let delta = position - query;
  return dot(delta, delta);
}
fn getRadiusKey(radius: f32) -> f32 { return radius * radius; }
fn getSeparationMeters(key: f32) -> f32 { return sqrt(key); }
fn getSquaredMeters(key: f32) -> f32 { return key; }
fn getPairMeters(a: vec2f, b: vec2f) -> f32 {
  let separation = b - a;
  return sqrt(dot(separation, separation));
}
fn isExactHit(key: f32, position: vec2f, query: vec2f) -> bool { return key == 0.0; }
fn getQueryBox(center: vec2f, radius: f32) -> QueryBox {
  return QueryBox(center - vec2f(radius), center + vec2f(radius));
}`;
  }
  const radius = getWGSLFloatLiteral(separation.sphereRadius);
  return /* wgsl */ `
${GEODESIC_WGSL}
const SPHERE_RADIUS: f32 = ${radius};
struct QueryBox { minimum: vec2f, maximum: vec2f, }
// The key is the haversine term h = sin^2(d / 2R): monotonic in distance, so ordering and radius
// tests need no inverse trigonometry.
fn getSeparationKey(position: vec2f, query: vec2f) -> f32 {
  return geodesicHaversineTerm(query, position);
}
fn getRadiusKey(radius: f32) -> f32 {
  let halfAngle = min(0.5 * radius / SPHERE_RADIUS, 1.5707963);
  let sine = sin(halfAngle);
  return sine * sine;
}
fn getSeparationMeters(key: f32) -> f32 {
  return 2.0 * SPHERE_RADIUS * asin(sqrt(clamp(key, 0.0, 1.0)));
}
fn getSquaredMeters(key: f32) -> f32 {
  let meters = getSeparationMeters(key);
  return meters * meters;
}
fn getPairMeters(a: vec2f, b: vec2f) -> f32 { return geodesicHaversineDistance(a, b, SPHERE_RADIUS); }
fn isExactHit(key: f32, position: vec2f, query: vec2f) -> bool { return all(position == query); }
// Conservative lon/lat box around the cap of the given radius: the latitude half-span is r / R,
// the longitude half-span is that divided by the cosine of the highest latitude the box reaches
// (1 percent slack for f32), and a box reaching a pole spans every longitude. No antimeridian
// wrap: a cap crossing +-180 is cut at the index domain edge.
fn getQueryBox(center: vec2f, radius: f32) -> QueryBox {
  let latitudeSpan = min(radius / SPHERE_RADIUS * 57.29577951308232 * 1.01, 180.0);
  let highestLatitude = min(abs(center.y) + latitudeSpan, 90.0);
  var longitudeSpan = 360.0;
  if (highestLatitude < 89.9) {
    longitudeSpan = min(latitudeSpan / geodesicCosLatitude(highestLatitude), 360.0);
  }
  let span = vec2f(longitudeSpan, latitudeSpan);
  return QueryBox(center - span, center + span);
}`;
}
