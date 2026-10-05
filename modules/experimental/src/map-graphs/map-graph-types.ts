// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView,
  GraphVectorView
} from '@luma.gl/gpgpu/gpu-core';

/**
 * A packed `uint32` view that may be one chunk or an ordered vector of chunks.
 *
 * Recipes accept both forms for source-aligned masks and IDs so they preserve the caller's batch
 * and chunk boundaries instead of repacking data.
 */
export type GPUMapGraphUint32Rows = GraphDataView<'uint32'> | GraphVectorView<'uint32'>;

/**
 * Packed two-dimensional f32 positions, for example projected map coordinates or
 * longitude/latitude pairs.
 */
export type GPUMapGraphPositions2D = GraphDataView<'float32x2'> | GraphVectorView<'float32x2'>;

/**
 * Caller-owned, capacity-bounded compact result shared by map-graph recipes.
 *
 * This shape is structurally identical to the geospatial `GPUSpatialQueryOutput`, so one result
 * can feed either family. `count` is always clamped to `ids.length` and is therefore safe to use as
 * an indirect draw instance count. Writable views must not alias each other or recipe inputs.
 */
export type GPUMapGraphCompactOutput = {
  /** Compact stable source IDs, or zero-based row indices when the recipe has no source IDs. */
  ids: GraphDataView<'uint32'>;
  /** One-row scalar receiving `min(totalCount, ids.length)`. */
  count: GraphDataView<'uint32'>;
  /** One-row scalar receiving `1` when any capacity in the recipe overflowed, otherwise `0`. */
  overflow: GraphDataView<'uint32'>;
  /** Optional one-row scalar receiving the unclamped number of accepted rows. */
  totalCount?: GraphDataView<'uint32'>;
};

/**
 * A prebuilt GPU Core command graph for one common map task.
 *
 * A recipe is an ordinary GPU Core contributor: it validates typed graph views in its constructor
 * and, when the caller runs `graph.add(recipe)`, returns command nodes for the caller-owned graph.
 * It never compiles, encodes, submits, or reads back. Inputs and outputs are `GraphDataView`,
 * `GraphVectorView`, or `GraphTextureView` instances that belong to the target graph. Values that
 * change every frame live in imported buffers that the application rewrites between encodings, so
 * updates never recompile the graph.
 *
 * @typeParam Parameters Per-encoding application parameters forwarded to node callbacks and CPU
 * conditions. Most recipes are generic over this type and never inspect it.
 */
export interface GPUMapGraphRecipe<Parameters = void> extends GPUCommandNodeProducer<Parameters> {
  /** Prefix for every graph node and transient-resource ID the recipe creates. */
  readonly id: string;
  /** Stable kebab-case recipe name, such as `'point-density'`, used for diagnostics and docs. */
  readonly recipe: string;
  /**
   * Returns this recipe's command nodes, creating any graph-owned transients on `graph`.
   *
   * Node IDs are `${id}-${step}` and are deterministic across calls with the same props.
   */
  getCommandNodes(graph: GPUCommandGraph<Parameters>): readonly GPUCommandNode<Parameters>[];
  /**
   * Releases GPU resources owned outside the graph.
   *
   * Only recipes that own such resources, such as a parameter buffer or a render model, implement
   * this method. Graph transients belong to the compiled graph and imported views to the caller.
   */
  destroy?(): void;
}
