// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import type {
  GPUNearestFeatureSource,
  GPUSpatialJoinGeometry,
  GPUSpatialJoinLines,
  GPUSpatialJoinPolygons
} from './spatial-join-types';

/** Value written to `neighborSegmentIndices` when the distance has no feature edge (a point feature, or containment). */
export const GPU_NEAREST_NO_SEGMENT = 0xffffffff;

/**
 * Feature geometry accepted by {@link GPUNearestFeatureJoin}: points, single segments, linestrings
 * and polygons or multipolygons (GeoArrow layout, see `GPUSpatialJoinPolygons`).
 */
export type GPUNearestFeatureGeometry =
  | GPUNearestFeatureSource
  | GPUSpatialJoinLines
  | GPUSpatialJoinPolygons;

/**
 * Query geometry for the k-nearest mode when it is not a chunked point set: points, linestrings or
 * polygons, one query feature per row. Line and polygon queries are not chunked.
 */
export type GPUNearestQueryGeometry = GPUSpatialJoinGeometry;

/**
 * Tie rule of the k-nearest mode.
 *
 * - `'lowest-id'`: exactly the `k` best `(distance, feature row)` pairs, so a tie at the k-th
 *   distance keeps the smallest feature rows.
 * - `'all'`: the `k` nearest plus every feature whose distance equals the k-th distance
 *   (pandas `sjoin_nearest(..., how='inner')` for `k = 1`). Needs `neighborCapacity > k` to
 *   hold more than `k` rows.
 */
export type GPUNearestTieMode = 'lowest-id' | 'all';

/** Internal description of one geometry side of a generated nearest kernel. @internal */
export type NearestSide = {
  /** WGSL identifier prefix: `q` for the query, `r` for the feature. */
  prefix: 'q' | 'r';
  /** Geometry kind. */
  kind: GPUNearestFeatureGeometry['kind'];
  /** Number of vertices in the position buffer, used to clamp malformed offsets. */
  vertexCount: number;
};

/** Packed rows used by a polygon side after ring ranges are precomputed. @internal */
export type NearestPolygonRings = GraphDataView<'uint32x2'>;
