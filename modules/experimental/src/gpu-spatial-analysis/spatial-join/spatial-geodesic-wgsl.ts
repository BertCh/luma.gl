// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GEODESIC_WGSL} from '../geometry-measures/geodesic-wgsl';

/**
 * Largest great-circle `dwithin` distance as a central angle in radians (72 degrees, about
 * 8000 km on the mean Earth sphere). Beyond it a point's gnomonic image of a far great-circle arc
 * is no longer reliable, so a larger distance selects nothing. @internal
 */
export const SPATIAL_GEODESIC_MAXIMUM_ANGLE = 1.2566;

/**
 * WGSL for the great-circle spatial-join kernels: the shared `GEODESIC_WGSL`, whose
 * `geodesicSegmentFraction(point, start, end)` and `geodesicSegmentDistance(point, start, end,
 * fraction, radius)` measure a point to a segment that is straight in longitude/latitude (the same
 * edge model as the planar-in-degrees containment and intersection tests, and as the network
 * snapping contributors). Passing `radius = 1.0` returns the central angle in radians.
 * Positions are `vec2f(longitude, latitude)` in degrees.
 *
 * @internal
 */
export const SPATIAL_GEODESIC_WGSL = GEODESIC_WGSL;
