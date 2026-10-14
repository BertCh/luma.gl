// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPU_EARTH_MEAN_RADIUS_METERS} from '../contracts/earth';
import {
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../contracts/spatial-context';

/**
 * Resolves the context of a movement operator with a great-circle path. Longitude/latitude
 * positions need an explicit `'great-circle'` metric: degrees are never treated as planar units.
 *
 * @internal
 */
export function resolveTrajectorySpatialContext(
  id: string,
  context: GPUSpatialContext | undefined
): GPUSpatialContext {
  const resolved = resolveGPUSpatialContext(
    id,
    context,
    GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT
  );
  if (resolved.coordinateSpace === 'longitude-latitude' && resolved.metric !== 'great-circle') {
    throw new Error(`${id} longitude-latitude positions need the great-circle metric`);
  }
  if (resolved.units === 'kilometers') {
    throw new Error(`${id} great-circle results are in meters; kilometers are not supported`);
  }
  return resolved;
}

/**
 * Sphere radius in meters when the context asks for great-circle geometry, otherwise `undefined`
 * (planar). Defaults to `GPU_EARTH_MEAN_RADIUS_METERS`.
 *
 * @internal
 */
export function getTrajectoryGeodesicRadius(context: GPUSpatialContext): number | undefined {
  return context.metric === 'great-circle'
    ? (context.sphereRadius ?? GPU_EARTH_MEAN_RADIUS_METERS)
    : undefined;
}
