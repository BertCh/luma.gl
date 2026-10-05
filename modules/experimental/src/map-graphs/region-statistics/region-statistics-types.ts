// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUGridIndexView, GraphDataView} from '@luma.gl/gpgpu/gpu-core';

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
  /** `output.ids` was too small for every selected row. */
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
  result: GraphDataView<'uint32'>;
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
 * selected IDs, and `outputMask` are bit-identical to the brute-force path, because the same exact
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
 * inputs, chunked index storage, and 3D indices. `output` and `outputMask` still cost O(rowCount) (mask clear and scatter, then the
 * unchanged visibility workflow); only candidate gathering, predicate, counts, reductions, and
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
  /** Whether `output.ids` was too small. */
  selectionTruncated: boolean;
  /** Whether the region input was truncated. */
  regionTruncated: boolean;
  /** Whether the grid candidate path may have missed rows. Always false without `spatialIndex`. */
  candidatesTruncated: boolean;
};
