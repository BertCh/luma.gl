// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView, GraphVectorView} from '@luma.gl/gpgpu/gpu-core';

/**
 * A packed `uint32` view that may be one chunk or an ordered vector of chunks.
 *
 * Contributors accept both forms for source-aligned masks and IDs so they preserve the caller's batch
 * and chunk boundaries instead of repacking data.
 */
export type GPUUint32Rows = GraphDataView<'uint32'> | GraphVectorView<'uint32'>;

/**
 * Packed two-dimensional f32 positions, for example projected map coordinates or
 * longitude/latitude pairs.
 */
export type GPUFloat32Positions = GraphDataView<'float32x2'> | GraphVectorView<'float32x2'>;

/**
 * Caller-owned, capacity-bounded compact result shared by GPU contributors.
 *
 * This shape is structurally identical to the geospatial `GPUSpatialQueryOutput`, so one result
 * can feed either family. `count` is always clamped to `ids.length` and is therefore safe to use as
 * an indirect draw instance count. Writable views must not alias each other or contributor inputs.
 */
export type GPUCompactOutput = {
  /** Compact stable source IDs, or zero-based row indices when the contributor has no source IDs. */
  ids: GraphDataView<'uint32'>;
  /** One-row scalar receiving `min(totalCount, ids.length)`. */
  count: GraphDataView<'uint32'>;
  /** One-row scalar receiving `1` when any capacity in the contributor overflowed, otherwise `0`. */
  overflow: GraphDataView<'uint32'>;
  /** Optional one-row scalar receiving the unclamped number of accepted rows. */
  totalCount?: GraphDataView<'uint32'>;
};
