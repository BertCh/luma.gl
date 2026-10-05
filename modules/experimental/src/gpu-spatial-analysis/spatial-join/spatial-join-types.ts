// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/gpgpu/gpu-core';

/** Value written to per-point feature outputs when a point joins no feature. */
export const GPU_SPATIAL_JOIN_NO_FEATURE = 0xffffffff;

/** Value written to `nearestDistances` when a point has no feature within the radius. */
export const GPU_SPATIAL_JOIN_NO_DISTANCE = -1;

/** Point features for `GPUNearestFeatureJoin`. */
export type GPUNearestFeaturePoints = {
  /** Feature source discriminator. */
  kind: 'points';
  /** Feature positions, one row per feature. Per-frame contents. */
  positions: GraphDataView<'float32x2'>;
};

/**
 * Closed line-segment features for `GPUNearestFeatureJoin`.
 *
 * Express a linestring as consecutive segment rows that share a `featureIds` value.
 */
export type GPUNearestFeatureSegments = {
  /** Feature source discriminator. */
  kind: 'segments';
  /** Segment starts, one row per feature. Per-frame contents. */
  starts: GraphDataView<'float32x2'>;
  /** Segment ends, one row per feature. Per-frame contents. */
  ends: GraphDataView<'float32x2'>;
};

/** Feature geometry accepted by `GPUNearestFeatureJoin`. */
export type GPUNearestFeatureSource = GPUNearestFeaturePoints | GPUNearestFeatureSegments;

/**
 * Point features for `GPUSpatialPredicateJoin`: one feature per `positions` row.
 */
export type GPUSpatialJoinPoints = {
  /** Geometry discriminator. */
  kind: 'points';
  /** Feature positions, one row per feature. Coordinates must be finite. */
  positions: GraphDataView<'float32x2'>;
};

/**
 * Linestring features for `GPUSpatialPredicateJoin`: one linestring per feature.
 *
 * A linestring needs at least two vertices; shorter features are empty and never match. A
 * linestring whose first and last vertex coincide is closed and has no boundary.
 */
export type GPUSpatialJoinLines = {
  /** Geometry discriminator. */
  kind: 'lines';
  /** Flattened vertices of all linestrings. */
  positions: GraphDataView<'float32x2'>;
  /** Feature-to-vertex offsets with `featureCount + 1` entries, first 0. */
  lineOffsets: GraphDataView<'uint32'>;
};

/**
 * Polygon or multipolygon features for `GPUSpatialPredicateJoin`, in the GeoArrow layout
 * of `GPUPointInPolygonJoin`.
 *
 * Rings close implicitly and need at least three vertices (shorter rings are ignored). Ring 0 of a
 * polygon is its shell and later rings are holes. Rings are expected to be valid in the OGC sense:
 * holes lie inside the shell, rings of one polygon do not cross, and polygons of one multipolygon
 * have disjoint interiors. Containment is evaluated with even/odd fill over all rings of a
 * feature, which is correct for valid input.
 */
export type GPUSpatialJoinPolygons = {
  /** Geometry discriminator. */
  kind: 'polygons';
  /** Flattened ring vertices. */
  positions: GraphDataView<'float32x2'>;
  /** Feature-to-polygon offsets with `featureCount + 1` entries, first 0. */
  featureOffsets: GraphDataView<'uint32'>;
  /** Polygon-to-ring offsets with a terminal entry. */
  polygonOffsets: GraphDataView<'uint32'>;
  /** Ring-to-vertex offsets with a terminal entry. */
  ringOffsets: GraphDataView<'uint32'>;
};

/** Feature geometry accepted on either side of `GPUSpatialPredicateJoin`. */
export type GPUSpatialJoinGeometry =
  | GPUSpatialJoinPoints
  | GPUSpatialJoinLines
  | GPUSpatialJoinPolygons;

/**
 * Caller-owned, capacity-bounded list of matched `(left, right)` feature rows.
 *
 * Slots `[0, count)` are sorted by `leftIds` and then `rightIds`, with no duplicates. `count` is
 * clamped to the capacity (`leftIds.length`), like `GPUCompactOutput`. Writable views must
 * not alias each other or the join inputs.
 */
export type GPUSpatialJoinPairs = {
  /** Left feature row per matched pair. Its length is the pair capacity. */
  leftIds: GraphDataView<'uint32'>;
  /** Right feature row per matched pair; same length as `leftIds`. */
  rightIds: GraphDataView<'uint32'>;
  /** One-row scalar receiving `min(totalCount, leftIds.length)`. */
  count: GraphDataView<'uint32'>;
  /** One-row scalar receiving `1` when any capacity overflowed, otherwise `0`. */
  overflow: GraphDataView<'uint32'>;
  /** Optional one-row scalar receiving the unclamped number of matched pairs. */
  totalCount?: GraphDataView<'uint32'>;
};
