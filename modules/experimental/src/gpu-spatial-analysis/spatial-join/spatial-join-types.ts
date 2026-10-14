// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import type {
  GPULineGeometryPort,
  GPUPackedGeometryPort,
  GPUPointGeometryPort,
  GPUPolygonGeometryPort
} from '../../analysis-core/contracts/index';

/** Value written to per-point feature outputs when a point joins no feature. */
export const GPU_SPATIAL_JOIN_NO_FEATURE = 0xffffffff;

/** Value written to `nearestDistances` when a point has no feature within the radius. */
export const GPU_SPATIAL_JOIN_NO_DISTANCE = -1;

/**
 * Closed line-segment features for `GPUNearestFeatureJoin`.
 *
 * Express a linestring as consecutive segment rows that share a `sourceIds` value.
 */
export type GPUNearestFeatureSegments = {
  /** Feature source discriminator. */
  kind: 'segments';
  /** Segment starts, one row per feature. Per-frame contents. */
  starts: GraphDataView<'float32x2'>;
  /** Segment ends, one row per feature. Per-frame contents. */
  ends: GraphDataView<'float32x2'>;
  /** Optional stable feature IDs written instead of feature rows. */
  sourceIds?: GraphDataView<'uint32'>;
};

/** Feature geometry accepted by `GPUNearestFeatureJoin`. */
export type GPUNearestFeatureSource =
  | GPUPackedGeometryPort<GPUPointGeometryPort>
  | GPUNearestFeatureSegments;

/**
 * Feature geometry accepted on either side of `GPUSpatialPredicateJoin`: a point, line or polygon
 * geometry port with one packed `positions` view.
 *
 * - Points: one feature per `positions` row. Coordinates must be finite.
 * - Lines: one linestring per feature. A linestring needs at least two vertices; shorter features
 *   are empty and never match. A linestring whose first and last vertex coincide is closed and has
 *   no boundary.
 * - Polygons or multipolygons (four-level GeoArrow layout): rings close implicitly and need at least
 *   three vertices (shorter rings are ignored). Ring 0 of a polygon is its shell and later rings are
 *   holes. Rings are expected to be valid in the OGC sense: holes lie inside the shell, rings of one
 *   polygon do not cross, and polygons of one multipolygon have disjoint interiors. Containment is
 *   evaluated with even/odd fill over all rings of a feature, which is correct for valid input.
 */
export type GPUSpatialJoinGeometry = GPUPackedGeometryPort<
  GPUPointGeometryPort | GPULineGeometryPort | GPUPolygonGeometryPort
>;

/**
 * Key columns for an attribute-equality join condition (GeoPandas `sjoin(on_attribute=...)`).
 *
 * A pair survives only when the left key equals the right key, on top of the spatial condition.
 * The keys are per-frame buffer contents; whether the option is present is compile-time.
 */
export type GPUSpatialJoinOnAttribute = {
  /** One `uint32` key per left row (per query row for `GPUNearestFeatureJoin`). */
  left: GraphDataView<'uint32'>;
  /** One `uint32` key per right row (per feature row for `GPUNearestFeatureJoin`). */
  right: GraphDataView<'uint32'>;
};
