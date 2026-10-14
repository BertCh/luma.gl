// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import type {GPUNearestFeatureSource} from './spatial-join-types';
import type {
  GPULineGeometryPort,
  GPUPackedGeometryPort,
  GPUPointGeometryPort,
  GPUPolygonGeometryPort
} from '../../analysis-core/contracts/index';

/** Value written to `neighborSegmentIndices` when the distance has no feature edge (a point feature, or containment). */
export const GPU_NEAREST_NO_SEGMENT = 0xffffffff;

/**
 * Feature geometry accepted by {@link GPUNearestFeatureJoin}: points, single segments, linestrings
 * and polygons or multipolygons (four-level layout). Optional `sourceIds` are written instead of
 * feature rows.
 */
export type GPUNearestFeatureGeometry =
  | GPUNearestFeatureSource
  | GPUPackedGeometryPort<GPULineGeometryPort>
  | GPUPackedGeometryPort<GPUPolygonGeometryPort>;

/**
 * Query geometry of {@link GPUNearestFeatureJoin}, one query per row: points (possibly chunked),
 * linestrings or polygons. Line and polygon queries are one packed view each and need neighbors mode.
 */
export type GPUNearestQueryGeometry =
  | GPUPointGeometryPort
  | GPUPackedGeometryPort<GPULineGeometryPort | GPUPolygonGeometryPort>;

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
  /** Word offset of ring offsets packed after the polygon feature ring ranges, when present. */
  packedRingOffsetsBase?: number;
};

/** Packed rows used by a polygon side after ring ranges are precomputed. @internal */
export type NearestPolygonRings = GraphDataView<'uint32x2'>;
