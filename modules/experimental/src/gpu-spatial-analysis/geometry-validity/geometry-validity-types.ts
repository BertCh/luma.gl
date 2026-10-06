// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Bits written to the per-feature mask of `GPUGeometryValidity`. A feature with mask `0` has none
 * of the listed problems.
 */
export const GPU_GEOMETRY_VALIDITY_BIT = {
  /** A ring has a vertex with a NaN or infinite coordinate. */
  nonFinite: 1 << 0,
  /** `ringClosure: 'explicit'` only: a ring's last vertex differs from its first. */
  unclosedRing: 1 << 1,
  /** A ring has fewer than three distinct vertices once a closing duplicate is dropped. */
  shortRing: 1 << 2,
  /** Two consecutive vertices of a ring are equal. */
  repeatedVertex: 1 << 3,
  /**
   * A ring meets itself: two non-adjacent edges of one ring cross, touch or overlap, or two
   * adjacent edges double back over each other (a spike).
   */
  selfIntersection: 1 << 4,
  /**
   * Edges of two different rings of the same feature cross properly or overlap along a segment,
   * which covers crossing holes, a hole crossing its shell, and overlapping polygons of a
   * multipolygon. Rings that touch at single points are allowed.
   */
  crossingRings: 1 << 5,
  /** A hole of a polygon lies outside, or on the boundary of, that polygon's shell. */
  holeOutsideShell: 1 << 6,
  /**
   * A ring winds the wrong way for the configured `orientation`. This is a convention rather than
   * a validity rule and is excluded from {@link GPU_GEOMETRY_VALIDITY_STRUCTURAL_MASK}.
   */
  badOrientation: 1 << 7,
  /** A predicate could not be certified (non-finite input or an extreme exponent range). */
  uncertain: 1 << 8
} as const;

/** Union of every bit that makes a polygon structurally invalid: all bits except `badOrientation`. */
export const GPU_GEOMETRY_VALIDITY_STRUCTURAL_MASK =
  GPU_GEOMETRY_VALIDITY_BIT.nonFinite |
  GPU_GEOMETRY_VALIDITY_BIT.unclosedRing |
  GPU_GEOMETRY_VALIDITY_BIT.shortRing |
  GPU_GEOMETRY_VALIDITY_BIT.repeatedVertex |
  GPU_GEOMETRY_VALIDITY_BIT.selfIntersection |
  GPU_GEOMETRY_VALIDITY_BIT.crossingRings |
  GPU_GEOMETRY_VALIDITY_BIT.holeOutsideShell |
  GPU_GEOMETRY_VALIDITY_BIT.uncertain;

/** Ring orientation convention checked by `GPUGeometryValidity`. */
export type GPUGeometryValidityOrientation =
  /** Shells wind counter-clockwise and holes clockwise (RFC 7946, OGC). */
  | 'counter-clockwise-shell'
  /** Shells wind clockwise and holes counter-clockwise (Shapefile, Mapbox vector tiles). */
  | 'clockwise-shell'
  /** Never set `badOrientation`. */
  | 'ignore';

/** How `GPUGeometryValidity` treats ring closure. */
export type GPUGeometryValidityRingClosure =
  /** Rings close implicitly (the repository convention); a repeated first vertex is tolerated. */
  | 'implicit'
  /** GeoJSON and GeoArrow style: a ring must end on its first vertex, else `unclosedRing` is set. */
  | 'explicit';
