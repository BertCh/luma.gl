// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/gpgpu/gpu-core';

/** Coordinate interpretation shared by the line contributors. */
export type GPULineCoordinateSystem =
  /** Planar positions (projected or tile-local units); straight segments, Euclidean lengths. */
  | 'planar'
  /**
   * Longitude/latitude in degrees on a sphere; segments are great-circle arcs and lengths are
   * central angle times the sphere radius.
   */
  | 'spherical';

/** Value written to `sourcePaths` / `sourceRows` rows that carry no source. */
export const GPU_LINE_NO_SOURCE = 0xffffffff;

/**
 * Caller-owned, capacity-bounded path output written by the path-emitting contributors
 * (`GPULineSegmentize`, `GPUGreatCircleArcs`, `GPULineSmooth`, `GPULineChunk`).
 *
 * Paths use the toolkit's flat layout: vertex rows in `positions`, path `p` owning rows
 * `[pathOffsets[p], pathOffsets[p + 1])`, which is what deck.gl `PathLayer` binary attributes
 * (`startIndices`) consume. The vertex capacity is `positions.length` (compile time).
 *
 * Every offset is clamped to the capacity, so when `overflow` is 1 the last paths are truncated
 * (possibly to zero vertices) but the layout stays valid. `count` is `min(total, capacity)` and is
 * safe as an indirect draw count. Every word is rewritten on every encoding.
 */
export type GPULinePathOutput = {
  /** Output vertices, `capacity` rows. Rows at or past `count` are unspecified. */
  positions: GraphDataView<'float32x2'>;
  /**
   * Path start offsets, `pathCapacity + 1` rows, clamped to the vertex capacity. For contributors that
   * emit exactly one path per input path or pair, `pathCapacity` equals the input path count.
   */
  pathOffsets: GraphDataView<'uint32'>;
  /** One-row scalar receiving the clamped vertex count. */
  count: GraphDataView<'uint32'>;
  /** One-row scalar receiving `1` when the vertex or path capacity overflowed, otherwise `0`. */
  overflow: GraphDataView<'uint32'>;
  /** Optional one-row scalar receiving the unclamped vertex count. */
  requiredCount?: GraphDataView<'uint32'>;
  /**
   * Optional one-row scalar receiving the clamped number of output paths. Required by contributors
   * whose path count is data dependent (chunking); otherwise equals the input path count.
   */
  pathCount?: GraphDataView<'uint32'>;
  /**
   * Optional per-output-path source path (or source pair) row, `pathCapacity` rows,
   * `GPU_LINE_NO_SOURCE` past `pathCount`.
   */
  sourcePaths?: GraphDataView<'uint32'>;
  /**
   * Optional per-output-vertex source segment row: the input row that starts the segment the
   * vertex was emitted on (the last vertex of a path maps to the path's last row).
   */
  sourceRows?: GraphDataView<'uint32'>;
  /**
   * Optional per-output-vertex measure: distance from the start of the source path along the
   * source geometry (planar units, or sphere-radius units, meters by default, for `'spherical'`).
   */
  measures?: GraphDataView<'float32'>;
};
