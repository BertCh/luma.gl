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
