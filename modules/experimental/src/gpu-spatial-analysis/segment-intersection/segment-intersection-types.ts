// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import type {
  GPUSpatialJoinLines,
  GPUSpatialJoinPairs,
  GPUSpatialJoinPolygons
} from '../spatial-join/index';

/**
 * Values written to `kinds` by `GPUSegmentIntersection`. Slot `0` never appears in the output.
 *
 * Classification uses exact orientation signs, so the kinds are exact for finite coordinates whose
 * products stay within 200 binary orders of magnitude of each other (subnormals are unsupported).
 */
export const GPU_SEGMENT_INTERSECTION_KIND = {
  /** The segment interiors cross at exactly one point. `points` holds the (rounded) crossing. */
  proper: 1,
  /**
   * The segments share exactly one point and the lines through them differ: an endpoint of one
   * lies on the other (a T-junction or a shared endpoint). `points` is that endpoint, exactly.
   */
  touch: 2,
  /** The segments are collinear and share exactly one endpoint. `points` is that endpoint. */
  collinearTouch: 3,
  /**
   * The segments are collinear and share a sub-segment of positive length. `points` and `endPoints`
   * are the two ends of the shared span, exactly, with `points` first along the lower coordinate.
   */
  overlap: 4,
  /**
   * The orientation of some triple could not be certified (non-finite input, or products spanning
   * more than 200 binary orders of magnitude). The pair may or may not intersect. `points` is the left start.
   */
  uncertain: 5
} as const;

/** One of the {@link GPU_SEGMENT_INTERSECTION_KIND} values. */
export type GPUSegmentIntersectionKind =
  (typeof GPU_SEGMENT_INTERSECTION_KIND)[keyof typeof GPU_SEGMENT_INTERSECTION_KIND];

/** Value written to segment ring, feature and successor columns that have no row. */
export const GPU_SEGMENT_NONE = 0xffffffff;

/**
 * Linestring or polygon geometry whose segments `GPUSegmentIntersection` intersects.
 *
 * The layouts are those of `GPUSpatialPredicateJoin`. A segment is identified by the index of its
 * start vertex in `positions` (its segment ID). Polygon rings close implicitly and rings with fewer
 * than three vertices are ignored; linestrings need two vertices and have no closing segment. A
 * zero-length segment (equal consecutive vertices) or a segment with a non-finite coordinate is
 * skipped and never intersects anything.
 */
export type GPUSegmentGeometry = GPUSpatialJoinLines | GPUSpatialJoinPolygons;

/**
 * Optional per-pair columns and scalars of `GPUSegmentIntersection`, all aligned with
 * `pairs.leftIds`. Slots `[0, pairs.count)` are valid.
 */
export type GPUSegmentIntersectionColumns = {
  /** Intersection kind per pair, a {@link GPU_SEGMENT_INTERSECTION_KIND} value. */
  kinds?: GraphDataView<'uint32'>;
  /** Intersection point per pair; the first end of the shared span for overlaps. */
  points?: GraphDataView<'float32x2'>;
  /** Second end of the shared span for overlaps; equal to `points` for every other kind. */
  endPoints?: GraphDataView<'float32x2'>;
  /** Feature row of the left segment. */
  leftFeatures?: GraphDataView<'uint32'>;
  /** Feature row of the right segment. */
  rightFeatures?: GraphDataView<'uint32'>;
  /** Global ring (polygons) or linestring (lines) row of the left segment. */
  leftRings?: GraphDataView<'uint32'>;
  /** Global ring or linestring row of the right segment. */
  rightRings?: GraphDataView<'uint32'>;
};

/** Output pair list; see {@link GPUSpatialJoinPairs}. Row IDs are segment IDs. */
export type GPUSegmentIntersectionPairs = GPUSpatialJoinPairs;
