// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUGridIndexView, GraphDataView} from '@luma.gl/gpgpu/gpu-core';
import type {GPUOverflowStatusPort} from '../../analysis-core/contracts/index';
import type {GPUCompactOutput} from '../../utils/gpu-contributor-types';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext,
  type GPUSpatialContextSupport
} from '../../analysis-core/contracts/spatial-context';

/** Number of `uint32` header words at the start of a region statistics summary. */
export const GPU_REGION_STATISTICS_HEADER_LENGTH = 8;

/** Word index of each field in a region statistics summary. Float fields are stored as f32 bits. */
export const GPU_REGION_STATISTICS_SUMMARY_LAYOUT = {
  /** `u32` rows inside the region. */
  selectedCount: 0,
  /** `u32` selected rows whose value is finite, 0 without values. */
  valueCount: 1,
  /** `f32` sum of finite selected values, 0 when `valueCount` is 0. */
  sum: 2,
  /** `f32` `sum / valueCount`, 0 when `valueCount` is 0. */
  mean: 3,
  /** `f32` minimum, 0 when `valueCount` is 0. */
  minimum: 4,
  /** `f32` maximum, 0 when `valueCount` is 0. */
  maximum: 5,
  /** `u32` `valueCount` minus the binned count, 0 without a histogram. */
  histogramOutsideCount: 6,
  /** `u32` bit field, see {@link GPU_REGION_STATISTICS_FLAGS}. */
  flags: 7,
  /** First of `binCount` `u32` histogram bins. */
  histogram: 8
} as const;

/** Bits of the summary `flags` word. */
export const GPU_REGION_STATISTICS_FLAGS = {
  /** `output.selectedRows.ids` was too small for every selected row. */
  selectionTruncated: 1,
  /** The region input was truncated: lasso vertex count above capacity, or pick-region overflow. */
  regionTruncated: 2,
  /**
   * Only with `spatialIndex`: the grid candidate query overflowed `candidateCapacity` or the index,
   * or the index holds fewer rows than the source (rows outside its domain or with non-finite
   * positions are missing). Results may then be incomplete.
   */
  candidatesTruncated: 4
} as const;

/** Rectangle `[minX, minY, maxX, maxY]`, inclusive, in world units or screen pixels. */
export type GPURegionRectangle = {
  /** Shape discriminator. */
  kind: 'rectangle';
  /** Per-frame packed `[minX, minY, maxX, maxY]`. */
  bounds: GraphDataView<'float32'>;
  /**
   * Optional per-frame 20-float screen transform: a column-major world-to-clip 4x4 matrix followed
   * by viewport width and height in pixels and two reserved zeros. When given, `bounds` is in
   * screen pixels with a top-left origin.
   */
  screenTransform?: GraphDataView<'float32'>;
};

/** Closed lasso polygon with even-odd fill and a per-frame vertex count. */
export type GPURegionPolygon = {
  /** Shape discriminator. */
  kind: 'polygon';
  /** Vertex storage. `vertices.length` is the compile-time vertex capacity. */
  vertices: GraphDataView<'float32x2'>;
  /** Per-frame one-row active vertex count. Values above capacity are clamped and flagged. */
  vertexCount: GraphDataView<'uint32'>;
  /** Optional per-frame 20-float screen transform. When given, `vertices` are in screen pixels. */
  screenTransform?: GraphDataView<'float32'>;
};

/** Region shapes evaluated by `GPURegionMask`. */
export type GPURegionShape = GPURegionRectangle | GPURegionPolygon;

/** World-space circle `[centerX, centerY, radius]`, inclusive. */
export type GPURegionRadius = {
  /** Shape discriminator. */
  kind: 'radius';
  /** Per-frame packed `[centerX, centerY, radius]`. */
  circle: GraphDataView<'float32'>;
};

/** Visible-object selection produced by `GPUIndexPickingTarget.addRegionPass`. */
export type GPURegionPickSelection = {
  /** Selection discriminator. */
  kind: 'pick-region';
  /** Packed region-pick result `[count, overflow, (objectIndex, batchIndex)...]`. */
  picks: GraphDataView<'uint32'>;
  /** Compile-time batch filter. Omit it to accept every batch. */
  batchIndex?: number;
};

/** A caller-computed source-aligned selection mask. Nonzero rows are selected. */
export type GPURegionMaskSelection = {
  /** Selection discriminator. */
  kind: 'mask';
  /** Source-aligned mask. */
  mask: GraphDataView<'uint32'>;
};

/**
 * Caller-owned outputs of `GPURegionMask` and `GPUPickRegionMask`: a source-aligned 0/1 mask and a
 * one-row truncation flag, both rewritten every encoding.
 */
export type GPURegionMaskOutput = {
  /** Packed mask with one row per source row. Every row is written with 0 or 1. */
  mask: GraphDataView<'uint32'>;
  /**
   * `overflow`: one row, 1 when the region input was truncated (lasso vertex count above capacity,
   * or a pick-region overflow), else 0.
   */
  status: GPUOverflowStatusPort;
};

/** Caller-owned outputs of `GPURegionStatistics`. */
export type GPURegionStatisticsOutput = {
  /** Packed summary of `8 + binCount` uint32 words, see {@link GPU_REGION_STATISTICS_SUMMARY_LAYOUT}. */
  summary: GraphDataView<'uint32'>;
  /** Optional 0/1 source-aligned selection mask for highlighting. Not allowed with `kind: 'mask'`. */
  mask?: GraphDataView<'uint32'>;
  /**
   * Optional bounded stable-ID result (the compact-output contract). `selectedRows.ids.length` must
   * be at least 1.
   */
  selectedRows?: GPUCompactOutput;
  /**
   * Optional packed one-row view that receives the clamped selected count, the same value as
   * `selectedRows.status.count` (`min(selected, selectedRows.ids.length)`), rewritten every encoding.
   * Typically the `instanceCount` word of an indirect draw record, imported as a 1-row view at an
   * element offset so an instanced draw of the selected rows needs no readback. Requires
   * `selectedRows`: the draw indexes rows through `selectedRows.ids`, so a contributor with no id
   * list has nothing to draw, and statistics-only contributors read the selected count from
   * `summary` instead. Must not alias any other output or input.
   */
  drawInstanceCount?: GraphDataView<'uint32'>;
};

/** Every region input accepted by `GPURegionStatistics`. */
export type GPURegionSelection =
  | GPURegionShape
  | GPURegionRadius
  | GPURegionPickSelection
  | GPURegionMaskSelection;

/**
 * Uniform grid index used by `GPURegionStatistics` to gather candidate rows instead of testing every
 * source row. All fields are compile-time.
 *
 * Identity guarantee: when the summary's `candidatesTruncated` flag is 0, `selectedCount`,
 * `valueCount`, `minimum`, `maximum`, `histogram`, `histogramOutsideCount`, the other flags, the
 * selected IDs, and `output.mask` are bit-identical to the brute-force path, because the same exact
 * predicate primitive runs on the candidates. `sum` and `mean` come from a reduction over a
 * compacted candidate order, so they are equal only up to float reassociation (exactly equal for
 * integer values with `|sum| < 2^24`).
 *
 * A lasso with a non-finite active vertex queries the whole index domain, so it only stays exact
 * while `candidateCapacity` covers every indexed row.
 *
 * Candidates are gathered from the contiguous `objectIds` ranges of the grid cells the region's
 * bounding box touches (plus a one-cell margin), so gather cost is O(gridHeight + candidateCapacity).
 * A non-finite circle radius selects nothing, as in brute force.
 *
 * Non-goals: screen-space selections, `pick-region` and `mask` selections, vector (chunked)
 * inputs, chunked index storage, and 3D indices. `output.selectedRows` and `output.mask` still
 * cost O(rowCount) (mask clear and scatter, then the unchanged visibility workflow); only candidate gathering, predicate, counts, reductions, and
 * histogram shrink to `candidateCapacity`.
 */
export type GPURegionStatisticsGridIndex = {
  /** Index discriminator. */
  kind: 'grid';
  /**
   * Grid storage and domain, typically a `GPUGridIndex` built over the same `positions` WITHOUT
   * `sourceIds` and with `firstSourceIndex` 0, so object IDs are source rows. The caller decides
   * when it is rebuilt: in the same graph (rebuilt every encoding, no speedup for fully dynamic
   * points) or in a separate graph that is encoded only when positions change. Must be
   * two-dimensional.
   */
  index: GPUGridIndexView;
  /**
   * Compile-time maximum candidate rows gathered per encoding. Clamped to the source row count.
   * Cells intersecting the region hold the candidates, so size it for the cells the region can
   * touch. Exceeding it sets `candidatesTruncated`.
   */
  candidateCapacity: number;
};

/** Fixed-bin histogram settings. */
export type GPURegionHistogramProps = {
  /** Compile-time bin count, an integer in `[1, 65536]`. */
  binCount: number;
  /**
   * Inclusive value domain. `'selection'` (default) uses the selected values' own GPU extent. A
   * literal pair is compile-time. A two-row `float32` view is per-frame.
   */
  domain?: 'selection' | readonly [number, number] | GraphDataView<'float32'>;
};

/** Decoded summary returned by `decodeGPURegionStatistics`. */
export type GPURegionStatisticsResult = {
  /** Rows inside the region. */
  selectedCount: number;
  /** Selected rows with a finite value. */
  valueCount: number;
  /** Sum of finite selected values. */
  sum: number;
  /** Mean of finite selected values. */
  mean: number;
  /** Minimum finite selected value. */
  minimum: number;
  /** Maximum finite selected value. */
  maximum: number;
  /** Bin counts, empty without a histogram. */
  histogram: Uint32Array;
  /** Finite selected values outside the histogram domain. */
  histogramOutsideCount: number;
  /** Whether `output.selectedRows.ids` was too small. */
  selectionTruncated: boolean;
  /** Whether the region input was truncated. */
  regionTruncated: boolean;
  /** Whether the grid candidate path may have missed rows. Always false without `spatialIndex`. */
  candidatesTruncated: boolean;
};

/**
 * Spatial contexts accepted by region selections. Rectangle, lasso and pick selections are
 * topological tests, so they work on planar coordinates or on longitude/latitude degrees (edges
 * are straight lines in degree space, with no antimeridian handling). Radius selections measure a
 * planar distance and reject longitude/latitude, see {@link resolveGPURegionSpatialContext}.
 */
export const GPU_REGION_SPATIAL_CONTEXT_SUPPORT: GPUSpatialContextSupport = Object.freeze({
  coordinateSpaces: Object.freeze(['planar', 'longitude-latitude'] as const),
  metrics: Object.freeze(['none', 'native'] as const)
});

/**
 * Validates a region operation's spatial context. A `'radius'` selection throws in
 * longitude/latitude because its radius is a planar distance in position units.
 * @internal
 */
export function resolveGPURegionSpatialContext(
  id: string,
  context: GPUSpatialContext | undefined,
  regionKind: GPURegionSelection['kind'] | undefined
): GPUSpatialContext {
  const resolved = resolveGPUSpatialContext(id, context, GPU_REGION_SPATIAL_CONTEXT_SUPPORT);
  if (regionKind === 'radius' && resolved.coordinateSpace === 'longitude-latitude') {
    // A radius is a planar distance in position units.
    throw new Error(`${id} radius regions need planar coordinates`);
  }
  return resolved;
}
