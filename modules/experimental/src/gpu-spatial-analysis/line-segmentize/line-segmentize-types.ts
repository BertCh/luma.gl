// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import type {GPUCardinalityPort, GPUOverflowStatusPort} from '../../analysis-core/contracts/index';

/**
 * Kernel-level coordinate interpretation derived from a `spatialContext` (not a user prop):
 * `'planar'` for planar contexts, `'spherical'` for longitude/latitude with a great-circle metric.
 */
export type GPULineCoordinateSystem =
  /** Planar positions (projected or tile-local units); straight segments, Euclidean lengths. */
  | 'planar'
  /**
   * Longitude/latitude in degrees on a sphere; segments are great-circle arcs and lengths are
   * central angle times the sphere radius.
   */
  | 'spherical';

/** Value written to `sourceLineIndices` / `sourceRows` rows that carry no source. */
export const GPU_LINE_NO_SOURCE = 0xffffffff;

/**
 * Caller-owned, capacity-bounded line output written by the line-emitting contributors
 * (`GPULineSegmentize`, `GPUGreatCircleArcs`, `GPULineSmooth`, `GPULineChunk`).
 *
 * Lines use the toolkit's flat layout: vertex rows in `positions`, line `p` owning rows
 * `[lineOffsets[p], lineOffsets[p + 1])`, which is what deck.gl `PathLayer` binary attributes
 * (`startIndices`) consume. The vertex capacity is `positions.length` (compile time).
 *
 * Every offset is clamped to the capacity, so when `status.overflow` is 1 the last lines are truncated
 * (possibly to zero vertices) but the layout stays valid. `status.count` is `min(total, capacity)` and is
 * safe as an indirect draw count. Every word is rewritten on every encoding.
 */
export type GPULineGeometryOutput = {
  /** Output vertices, `capacity` rows. Rows at or past `status.count` are unspecified. */
  positions: GraphDataView<'float32x2'>;
  /**
   * Line start offsets, `lineCapacity + 1` rows, clamped to the vertex capacity. For contributors that
   * emit exactly one line per input line or pair, `lineCapacity` equals the input line count.
   */
  lineOffsets: GraphDataView<'uint32'>;
  /**
   * One-row status scalars: `count` receives the clamped vertex count, `overflow` receives `1` when
   * the vertex or line capacity overflowed (otherwise `0`), the optional `requiredCount` receives
   * the unclamped vertex count, and the optional `lineCount` receives the clamped number of output
   * lines (required by contributors whose line count is data dependent, such as chunking;
   * otherwise equal to the input line count).
   */
  status: GPUCardinalityPort &
    GPUOverflowStatusPort & {
      lineCount?: GraphDataView<'uint32'>;
    };
  /**
   * Optional per-output-line source line (or source pair) row, `lineCapacity` rows,
   * `GPU_LINE_NO_SOURCE` past `status.lineCount`.
   */
  sourceLineIndices?: GraphDataView<'uint32'>;
  /**
   * Optional per-output-vertex source segment row: the input row that starts the segment the
   * vertex was emitted on (the last vertex of a line maps to the line's last row).
   */
  sourceRows?: GraphDataView<'uint32'>;
  /**
   * Optional per-output-vertex measure: distance from the start of the source line along the
   * source geometry (planar units, or sphere-radius units, meters by default, for `'spherical'`).
   */
  measures?: GraphDataView<'float32'>;
};
