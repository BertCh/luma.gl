// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUReduction,
  validatePackedUint32View,
  validatePackedView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/gpgpu/gpu-core';
import {createWGSLKernelNode, getWGSLFloatLiteral} from '../../utils/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../utils/gpu-contributor-utils';
import {getSlotGroupingNodes} from './slot-grouping';
import {type GPUSpatialWeights, validateGPUSpatialWeights} from './spatial-weights';

const OPERATION = 'GPUSpatialWeightsTranspose';

/** Properties for {@link GPUSpatialWeightsTranspose}. */
export type GPUSpatialWeightsTransposeProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'spatial-weights-transpose'`. */
  id?: string;
  /**
   * Weights `W` to transpose. Rows are the `offsets.length - 1` focus rows; neighbor IDs are
   * column indices in `[0, columnCount)`. Slots with a neighbor ID `>= columnCount` are dropped.
   */
  weights: GPUSpatialWeights;
  /**
   * Number of columns of `W` (rows of the transpose). Defaults to the row count of `weights`
   * (square weights). Set it for rectangular cross weights, whose neighbor IDs index the target
   * rows. Rows of the transpose that no column of `W` reaches are empty.
   */
  columnCount?: number;
  /**
   * Caller-owned CSR of `W'`: `offsets` holds `columnCount + 1` values, and `neighbors`,
   * `weights` (and `distances` when `weights.distances` is set) hold at least as many slots as the
   * input. Row `j` lists the source rows `i` with `w_ij` present, ascending.
   */
  output: GPUSpatialWeights;
  /**
   * Optional one-uint32 output: the number of positions `(i, j)` of the union of the patterns of
   * `W` and `W'` whose weights differ by more than `symmetryTolerance` or that are present in one
   * only. Zero means `W = W'`. Requires square weights.
   */
  asymmetricSlots?: GraphDataView<'uint32'>;
  /** Absolute weight tolerance of the symmetry check. Defaults to 0 (exact). */
  symmetryTolerance?: number;
};

/**
 * CSR transpose of a {@link GPUSpatialWeights} on the GPU (for square or rectangular weights),
 * the sparse `W.T` that libpysal `Graph` gets from swapping focal and neighbor.
 *
 * The result is deterministic and independent of any atomics ordering: slots are keyed by their
 * column and stable-sorted (so within a column they stay in source row order, which makes every
 * output row ascending), the output offsets are one binary search per column over the sorted keys
 * (the sort already counted, so there is no atomic count or scan), and a gather writes the
 * transposed slots. Capacity is the input
 * capacity (`nnz` cannot grow); slots of the output past `offsets[columnCount]` are written as
 * zero. `distances` are carried along when present.
 *
 * Optionally reports whether `W` is symmetric (`asymmetricSlots`).
 */
export class GPUSpatialWeightsTranspose implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSpatialWeightsTransposeProps;
  /** Rows of `W` (columns of the transpose). */
  readonly rowCount: number;
  /** Columns of `W` (rows of the transpose). */
  readonly columnCount: number;

  constructor(props: GPUSpatialWeightsTransposeProps) {
    const id = props.id ?? 'spatial-weights-transpose';
    this.id = id;
    this.props = props;
    const {weights, output} = props;
    const rows = validateGPUSpatialWeights(id, weights);
    const columns = props.columnCount ?? rows;
    if (!Number.isInteger(columns) || columns < 1) {
      throw new Error(`${id} columnCount must be a positive integer`);
    }
    const outputRows = validateGPUSpatialWeights(id, output, 'output');
    if (outputRows !== columns) {
      throw new Error(`${id} output.offsets must hold columnCount + 1 entries`);
    }
    if (output.neighbors.length < weights.neighbors.length) {
      throw new Error(`${id} output slot capacity must be at least the input slot capacity`);
    }
    if (weights.distances && !output.distances) {
      throw new Error(`${id} output.distances is required when weights.distances is set`);
    }
    if (props.asymmetricSlots) {
      validatePackedUint32View(props.asymmetricSlots, `${id} asymmetricSlots`);
      if (props.asymmetricSlots.length !== 1) {
        throw new Error(`${id} asymmetricSlots must hold one uint32 row`);
      }
      if (rows !== columns) {
        throw new Error(`${id} asymmetricSlots requires square weights`);
      }
    }
    if (
      props.symmetryTolerance !== undefined &&
      !(Number.isFinite(props.symmetryTolerance) && props.symmetryTolerance >= 0)
    ) {
      throw new Error(`${id} symmetryTolerance must be a non-negative finite number`);
    }
    if (output.distances) {
      validatePackedView(output.distances, ['float32'], `${id} output.distances`);
    }
    this.rowCount = rows;
    this.columnCount = columns;
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.offsets, output.neighbors, output.weights, output.distances, props.asymmetricSlots],
      [weights.offsets, weights.neighbors, weights.weights, weights.distances]
    );
  }

  /** Returns the transpose nodes (and the symmetry check when requested) in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, rowCount: rows, columnCount: columns} = this;
    const {weights, output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      weights.offsets,
      weights.neighbors,
      weights.weights,
      weights.distances,
      output.offsets,
      output.neighbors,
      output.weights,
      output.distances,
      props.asymmetricSlots
    ]);
    const capacity = weights.neighbors.length;
    const transient = <Format extends 'uint32' | 'float32'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, `${id}-${name}`, format, length);
    const slotKeys = transient('slot-keys', 'uint32', capacity);
    const slotIndices = transient('slot-indices', 'uint32', capacity);
    const constants = `const ROWS: u32 = ${rows}u;
const COLUMNS: u32 = ${columns}u;`;
    const grouping = getSlotGroupingNodes<Parameters>(graph, {
      id,
      operation: OPERATION,
      slotKeys,
      slotIndices,
      columns,
      columnOffsets: output.offsets
    });
    const {sortedKeys, sortedSlots} = grouping;
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-slot-keys`,
        operation: OPERATION,
        variant: 'slot-keys',
        bindings: [
          {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
          {name: 'slotKeys', view: slotKeys, type: 'u32', access: 'read_write'},
          {name: 'slotIndices', view: slotIndices, type: 'u32', access: 'read_write'}
        ],
        invocationCount: capacity,
        declarations: constants,
        body: `let neighbor = neighbors[neighborsOffset + index];
  let used = index < offsets[offsetsOffset + ROWS] && neighbor < COLUMNS;
  // Unused and out-of-range slots share the key COLUMNS, which sorts after every column.
  slotKeys[slotKeysOffset + index] = select(COLUMNS, neighbor, used);
  slotIndices[slotIndicesOffset + index] = index;`
      }),
      ...grouping.nodes,
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-gather`,
        operation: OPERATION,
        variant: weights.distances ? 'gather-distances' : 'gather',
        bindings: [
          {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
          {name: 'weights', view: weights.weights, type: 'f32', access: 'read'},
          {name: 'sortedKeys', view: sortedKeys, type: 'u32', access: 'read'},
          {name: 'sortedSlots', view: sortedSlots, type: 'u32', access: 'read'},
          {name: 'outputNeighbors', view: output.neighbors, type: 'u32', access: 'read_write'},
          {name: 'outputWeights', view: output.weights, type: 'f32', access: 'read_write'},
          ...(weights.distances
            ? [
                {name: 'distances', view: weights.distances, type: 'f32', access: 'read'} as const,
                {
                  name: 'outputDistances',
                  view: output.distances!,
                  type: 'f32',
                  access: 'read_write'
                } as const
              ]
            : [])
        ],
        invocationCount: output.neighbors.length,
        declarations: `${constants}
const INPUT_CAPACITY: u32 = ${capacity}u;

// Row of a used slot: the last row whose offset is at most the slot.
fn findRow(slot: u32) -> u32 {
  var low = 0u;
  var high = ROWS;
  while (low < high) {
    let middle = (low + high) / 2u;
    if (offsets[offsetsOffset + middle + 1u] <= slot) {
      low = middle + 1u;
    } else {
      high = middle;
    }
  }
  return low;
}`,
        body: `var neighbor = 0u;
  var weight = 0.0;
  var distance = 0.0;
  if (index < INPUT_CAPACITY && sortedKeys[sortedKeysOffset + index] < COLUMNS) {
    let slot = sortedSlots[sortedSlotsOffset + index];
    neighbor = findRow(slot);
    weight = weights[weightsOffset + slot];
    ${weights.distances ? 'distance = distances[distancesOffset + slot];' : ''}
  }
  outputNeighbors[outputNeighborsOffset + index] = neighbor;
  outputWeights[outputWeightsOffset + index] = weight;
  ${weights.distances ? 'outputDistances[outputDistancesOffset + index] = distance;' : ''}`
      })
    ];
    if (props.asymmetricSlots) {
      const rowMismatches = transient('row-mismatches', 'uint32', rows);
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-symmetry-rows`,
          operation: OPERATION,
          variant: 'symmetry-rows',
          bindings: [
            {name: 'offsets', view: weights.offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: weights.neighbors, type: 'u32', access: 'read'},
            {name: 'weights', view: weights.weights, type: 'f32', access: 'read'},
            {name: 'transposeOffsets', view: output.offsets, type: 'u32', access: 'read'},
            {name: 'transposeNeighbors', view: output.neighbors, type: 'u32', access: 'read'},
            {name: 'transposeWeights', view: output.weights, type: 'f32', access: 'read'},
            {name: 'rowMismatches', view: rowMismatches, type: 'u32', access: 'read_write'}
          ],
          invocationCount: rows,
          declarations: `${constants}
const TOLERANCE: f32 = ${getWGSLFloatLiteral(props.symmetryTolerance ?? 0)};
const NONE: u32 = 0xffffffffu;`,
          body: `var first = offsets[offsetsOffset + index];
  let firstEnd = offsets[offsetsOffset + index + 1u];
  var second = transposeOffsets[transposeOffsetsOffset + index];
  let secondEnd = transposeOffsets[transposeOffsetsOffset + index + 1u];
  var mismatches = 0u;
  loop {
    // Skip out-of-range neighbors of W; the transpose never holds them.
    while (first < firstEnd && neighbors[neighborsOffset + first] >= COLUMNS) {
      first++;
    }
    if (first >= firstEnd && second >= secondEnd) {
      break;
    }
    let firstColumn = select(NONE, neighbors[neighborsOffset + first], first < firstEnd);
    let secondColumn = select(
      NONE,
      transposeNeighbors[transposeNeighborsOffset + second],
      second < secondEnd
    );
    if (firstColumn == secondColumn) {
      if (abs(weights[weightsOffset + first] - transposeWeights[transposeWeightsOffset + second]) > TOLERANCE) {
        mismatches++;
      }
      first++;
      second++;
    } else if (firstColumn < secondColumn) {
      mismatches++;
      first++;
    } else {
      mismatches++;
      second++;
    }
  }
  rowMismatches[rowMismatchesOffset + index] = mismatches;`
        }),
        ...new GPUReduction({
          id: `${id}-symmetry-total`,
          input: rowMismatches,
          output: props.asymmetricSlots,
          operation: 'sum'
        }).getCommandNodes(graph)
      );
    }
    return nodes;
  }
}
