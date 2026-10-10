// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {VertexFormat} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import type {GPUCellTable} from '../cell-aggregation/index';
import type {GPURecipeStatusPort} from '../contracts/index';
import type {GPUSpatialWeights} from '../spatial-weights/index';

/**
 * The only caller-ownership override categories accepted by recipes.
 *
 * `inputs` reuses a caller-owned value produced before the recipe, `outputs` publishes terminal
 * values, and `scratch` exposes storage that only connects recipe stages. Algorithm parameters
 * remain ordinary recipe properties because they are not ownership overrides.
 */
export type GPURecipeOverrides<
  Inputs extends object = Record<never, never>,
  Outputs extends object = Record<never, never>,
  Scratch extends object = Record<never, never>
> = {
  inputs?: Inputs;
  outputs?: Outputs;
  scratch?: Scratch;
};

/** Fields every recipe result carries. */
export type GPURecipeResult<
  Outputs extends Record<string, unknown> = Record<string, unknown>,
  Intermediates extends Record<string, unknown> = Record<string, unknown>
> = {
  /** Contributors the recipe added to the graph, in declaration (execution) order. */
  contributors: readonly GPUCommandNodeProducer<never>[];
  /** Stable named terminal outputs. */
  outputs: Outputs;
  /** Useful named ports between stages; graph-owned views are not readback-capable. */
  intermediates: Intermediates;
  /** GPU-resident completeness and convergence views, grouped by the responsible stage. */
  status: GPURecipeStatusPort;
};

/**
 * Collects contributors, adds each to `graph` in call order and returns them.
 * Recipes use this so the order of `graph.add` calls is the order of the chain.
 * @internal
 */
export class RecipeBuilder<Parameters> {
  readonly contributors: GPUCommandNodeProducer<Parameters>[] = [];

  constructor(readonly graph: GPUCommandGraph<Parameters>) {}

  /** Adds `contributor` (its `getCommandNodes(graph)` runs now) and returns it. */
  add<Contributor extends GPUCommandNodeProducer<Parameters>>(
    contributor: Contributor
  ): Contributor {
    this.graph.add(contributor);
    this.contributors.push(contributor);
    return contributor;
  }
}

/**
 * Returns `provided` when given, else a graph-owned transient view. Transient views carry data
 * between nodes but cannot be read back, so pass a caller-owned view for every result you need.
 * @internal
 */
export function getOrCreateView<Format extends VertexFormat, Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  format: Format,
  length: number,
  provided?: GraphDataView<Format>
): GraphDataView<Format> {
  return provided ?? createTransientView(graph, id, format, length);
}

/** Creates a graph-owned transient CSR with `rows + 1` offsets and `capacity` slots. @internal */
export function createTransientSpatialWeights<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  rows: number,
  capacity: number,
  provided?: Partial<GPUSpatialWeights>
): GPUSpatialWeights {
  return {
    offsets: getOrCreateView(graph, `${id}-offsets`, 'uint32', rows + 1, provided?.offsets),
    neighbors: getOrCreateView(graph, `${id}-neighbors`, 'uint32', capacity, provided?.neighbors),
    weights: getOrCreateView(graph, `${id}-weights`, 'float32', capacity, provided?.weights),
    ...(provided?.distances ? {distances: provided.distances} : {})
  };
}

/**
 * Adapter: turns a cell table into the per-row analysis columns the statistics read.
 *
 * `values[i]` is `sumValues[i]` when the table has it, else the row count as f32. `mask[i]` is 1
 * for occupied rows and 0 for the empty tail of the capacity-bounded table.
 * @internal
 */
export function addCellTableColumnsNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  table: Pick<GPUCellTable, 'counts' | 'sumValues'>,
  output: {values: GraphDataView<'float32'>; mask: GraphDataView<'uint32'>}
): void {
  const rows = table.counts.length;
  graph.add(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-cell-table-columns`,
      operation: 'GPURecipeCellTableColumns',
      bindings: [
        {name: 'counts', view: table.counts, type: 'u32', access: 'read'},
        ...(table.sumValues
          ? [{name: 'sumValues', view: table.sumValues, type: 'f32', access: 'read'} as const]
          : []),
        {name: 'columnValues', view: output.values, type: 'f32', access: 'read_write'},
        {name: 'columnMask', view: output.mask, type: 'u32', access: 'read_write'}
      ],
      invocationCount: rows,
      body: `let rowCount = counts[countsOffset + index];
  columnValues[columnValuesOffset + index] = ${
    table.sumValues ? 'sumValues[sumValuesOffset + index]' : 'f32(rowCount)'
  };
  columnMask[columnMaskOffset + index] = select(0u, 1u, rowCount > 0u);`
    })
  );
}

/** Throws a recipe-prefixed error when `condition` is false. @internal */
export function assertRecipe(id: string, condition: unknown, message: string): asserts condition {
  if (!condition) {
    throw new Error(`${id} ${message}`);
  }
}
