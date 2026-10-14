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

/** One GPU-written unsigned scalar: a one-row `uint32` status word. */
export type GPUStatusScalar = GraphDataView<'uint32'>;

/**
 * Usable and, when known, exact required cardinality of a bounded result. Defined here because
 * `utils` sits below `analysis-core`; the analysis status contracts re-export it.
 */
export type GPUCardinalityPort = {
  /** Number of usable rows or elements written. */
  count: GPUStatusScalar;
  /** Exact unclamped requirement. Omit when the operation cannot compute it exactly. */
  requiredCount?: GPUStatusScalar;
};

/** Final-output completeness for a capacity-bounded result. */
export type GPUOverflowStatusPort = {
  /** One when the final result is incomplete because its output capacity was too small. */
  overflow: GPUStatusScalar;
};

/**
 * Caller-owned, capacity-bounded compact result shared by every GPU contributor and geospatial
 * query (the one compact-output contract; re-exported from the analysis contracts).
 *
 * `status.count` is always clamped to `ids.length` and is therefore safe to use as an indirect
 * draw instance count. Writable views must not alias each other or contributor inputs.
 */
export type GPUCompactOutput = {
  /** Compact stable source IDs, or zero-based row indices when the contributor has no source IDs. */
  ids: GraphDataView<'uint32'>;
  /**
   * One-row status scalars: `count` receives `min(requiredCount, ids.length)`, `overflow` receives
   * `1` when any capacity in the contributor overflowed (otherwise `0`), and the optional
   * `requiredCount` receives the unclamped number of accepted rows.
   */
  status: GPUCardinalityPort & GPUOverflowStatusPort;
};
