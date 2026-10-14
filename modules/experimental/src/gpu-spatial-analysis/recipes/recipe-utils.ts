// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {VertexFormat} from '@luma.gl/core';
import {
  createTransientView,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import type {GPUCellFamily, GPUCellTable} from '../cell-aggregation/index';
import {GPUCellTopology, getCellTopologyStride} from '../cell-topology/index';
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

/**
 * WGSL for a binary search of a 64-bit cell key in a sorted table of little-endian `(low, high)`
 * `uint32x2` rows, as written by `GPUCellAggregation`. Defines `${functionName}(key: vec2u) -> u32`
 * with `key = vec2u(high, low)`, returning the row or `0xffffffffu` when the key is absent.
 * `tableName` must be a bound `u32` array (with its generated `${tableName}Offset` constant) and
 * `countExpression` a WGSL `u32` expression for the number of occupied leading rows.
 * @internal
 */
export function getCellKeySearchWGSL(
  functionName: string,
  tableName: string,
  countExpression: string
): string {
  return `fn ${functionName}(key: vec2u) -> u32 {
  let rowCount = ${countExpression};
  var low = 0u;
  var high = rowCount;
  loop {
    if (low >= high) {
      break;
    }
    let middle = low + (high - low) / 2u;
    let middleLow = ${tableName}[${tableName}Offset + 2u * middle];
    let middleHigh = ${tableName}[${tableName}Offset + 2u * middle + 1u];
    if (middleHigh < key.x || (middleHigh == key.x && middleLow < key.y)) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  if (low < rowCount && ${tableName}[${tableName}Offset + 2u * low] == key.y && ${tableName}[${tableName}Offset + 2u * low + 1u] == key.x) {
    return low;
  }
  return 0xffffffffu;
}`;
}

/** Properties of {@link addCellDiskWeightsNodes}. @internal */
export type CellDiskWeightsProps = {
  family: GPUCellFamily;
  /** Sorted, unique study cell keys (`uint32x2` little-endian rows); occupied rows lead. */
  cells: GraphDataView<'uint32x2'>;
  /** One-row scalar with the number of occupied leading rows of `cells`. */
  count: GraphDataView<'uint32'>;
  /** Per-row mask for the topology: nonzero rows get a neighborhood. */
  mask: GraphDataView<'uint32'>;
  /** Grid distance of the disk, `k >= 1`. */
  rings: number;
  /** Output CSR (`rows + 1` offsets, `neighborCapacity` slots). */
  weights: GPUSpatialWeights;
  /** One-row scalar receiving 1 when `neighborCapacity` truncated the CSR. */
  overflow: GraphDataView<'uint32'>;
};

/**
 * Builds a binary topological CSR over a sorted cell table: `GPUCellTopology` disk of radius
 * `rings`, a binary search of every disk key in the table, a count pass, an inclusive scan, and a
 * write pass that stores each row's neighbor indices (self excluded, ascending) with weight 1.
 * Neighbors absent from the table are dropped. When the total exceeds the slot capacity the
 * offsets are clamped to it and `overflow` is set. Returns the topology stride.
 * @internal
 */
export function addCellDiskWeightsNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  builder: RecipeBuilder<Parameters>,
  id: string,
  props: CellDiskWeightsProps
): number {
  const {cells, weights} = props;
  const rows = cells.length;
  const capacity = weights.neighbors.length;
  const topology = builder.add(
    new GPUCellTopology({
      id: `${id}-disk`,
      family: props.family,
      operation: {type: 'disk', k: props.rings},
      cells,
      mask: props.mask,
      output: {
        cells: createTransientView(
          graph,
          `${id}-disk-cells`,
          'uint32x2',
          rows * getCellTopologyStride(props.family, {type: 'disk', k: props.rings})
        )
      }
    })
  );
  const stride = topology.stride;
  const diskCells = topology.props.output.cells;
  const counts = createTransientView(graph, `${id}-slot-counts`, 'uint32', rows);
  const inclusive = createTransientView(graph, `${id}-slot-ends`, 'uint32', rows);
  const declarations = `const STRIDE: u32 = ${stride}u;
const ROWS: u32 = ${rows}u;
const CAPACITY: u32 = ${capacity}u;
const NOT_FOUND: u32 = 0xffffffffu;`;
  const searchDeclarations = `${declarations}
${getCellKeySearchWGSL('findStudyCell', 'cells', 'countScalar[countScalarOffset]')}`;
  // Reads disk entry `entry` of this row as vec2u(high, low); zero entries are padding.
  const readEntry = `let entryLow = disk[diskOffset + 2u * (index * STRIDE + entry)];
    let entryHigh = disk[diskOffset + 2u * (index * STRIDE + entry) + 1u];
    if (entryLow == 0u && entryHigh == 0u) {
      continue;
    }
    let found = findStudyCell(vec2u(entryHigh, entryLow));
    if (found == NOT_FOUND || found == index) {
      continue;
    }`;
  graph.add(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-slot-counts`,
      operation: 'GPUCellDiskWeights',
      variant: 'count',
      bindings: [
        {name: 'disk', view: diskCells, type: 'u32', access: 'read'},
        {name: 'cells', view: cells, type: 'u32', access: 'read'},
        {name: 'countScalar', view: props.count, type: 'u32', access: 'read'},
        {name: 'slotCounts', view: counts, type: 'u32', access: 'read_write'}
      ],
      invocationCount: rows,
      declarations: searchDeclarations,
      body: `var count = 0u;
  for (var entry = 0u; entry < STRIDE; entry++) {
    ${readEntry}
    count++;
  }
  slotCounts[slotCountsOffset + index] = count;`
    })
  );
  graph.add(
    new GPUScan({id: `${id}-slot-scan`, input: counts, output: inclusive, mode: 'inclusive'})
  );
  graph.add(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-offsets`,
      operation: 'GPUCellDiskWeights',
      variant: 'offsets',
      bindings: [
        {name: 'slotEnds', view: inclusive, type: 'u32', access: 'read'},
        {name: 'rowOffsets', view: weights.offsets, type: 'u32', access: 'read_write'},
        {name: 'overflow', view: props.overflow, type: 'u32', access: 'read_write'}
      ],
      invocationCount: rows + 1,
      declarations,
      body: `var rowOffset = 0u;
  if (index > 0u) {
    rowOffset = min(slotEnds[slotEndsOffset + index - 1u], CAPACITY);
  }
  rowOffsets[rowOffsetsOffset + index] = rowOffset;
  if (index == ROWS) {
    overflow[overflowOffset] = select(0u, 1u, slotEnds[slotEndsOffset + ROWS - 1u] > CAPACITY);
  }`
    })
  );
  graph.add(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-write`,
      operation: 'GPUCellDiskWeights',
      variant: 'write',
      bindings: [
        {name: 'disk', view: diskCells, type: 'u32', access: 'read'},
        {name: 'cells', view: cells, type: 'u32', access: 'read'},
        {name: 'countScalar', view: props.count, type: 'u32', access: 'read'},
        {name: 'slotEnds', view: inclusive, type: 'u32', access: 'read'},
        {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read_write'},
        {name: 'weightValues', view: weights.weights, type: 'f32', access: 'read_write'}
      ],
      invocationCount: rows,
      declarations: searchDeclarations,
      body: `var start = 0u;
  if (index > 0u) {
    start = slotEnds[slotEndsOffset + index - 1u];
  }
  var written = 0u;
  for (var entry = 0u; entry < STRIDE; entry++) {
    ${readEntry}
    let slot = start + written;
    if (slot < CAPACITY) {
      neighbors[neighborsOffset + slot] = found;
      weightValues[weightValuesOffset + slot] = 1.0;
    }
    written++;
  }
  // Ascending neighbor index within the row (insertion sort of the slots that fit).
  let stored = select(0u, min(written, CAPACITY - start), start < CAPACITY);
  for (var i = 1u; i < stored; i++) {
    let value = neighbors[neighborsOffset + start + i];
    var j = i;
    loop {
      if (j == 0u || neighbors[neighborsOffset + start + j - 1u] <= value) {
        break;
      }
      neighbors[neighborsOffset + start + j] = neighbors[neighborsOffset + start + j - 1u];
      j--;
    }
    neighbors[neighborsOffset + start + j] = value;
  }`
    })
  );
  return stride;
}
